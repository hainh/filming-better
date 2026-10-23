//! Output muxer: encoded video + stream-copied audio / subtitles, chapters and metadata.
//!
//! The output is written to `<stem>.fbtmp<.ext>` and renamed to the final name once the
//! trailer has been written, so an interrupted run never leaves a half file under the real name.

use super::decode::SidePacket;
use super::encode::{VideoCodec, VideoEncoder};
use super::probe::*;
use super::*;
use std::path::{Path, PathBuf};

const SUB_TB: AVRational = AVRational { num: 1, den: 1000 };
const TB_US: AVRational = AVRational { num: 1, den: 1_000_000 };

/// Decodes text subtitles and re-encodes them in a format the output container accepts.
pub struct SubTranscoder {
    dec: *mut ff::AVCodecContext,
    enc: *mut ff::AVCodecContext,
    buf: Vec<u8>,
}
unsafe impl Send for SubTranscoder {}

fn is_text_sub(id: ff::AVCodecID) -> bool {
    [
        ff::AV_CODEC_ID_SUBRIP,
        ff::AV_CODEC_ID_SRT,
        ff::AV_CODEC_ID_ASS,
        ff::AV_CODEC_ID_SSA,
        ff::AV_CODEC_ID_MOV_TEXT,
        ff::AV_CODEC_ID_WEBVTT,
        ff::AV_CODEC_ID_TEXT,
    ]
    .contains(&id)
}

impl SubTranscoder {
    fn target_name(&self) -> String {
        unsafe { std::ffi::CStr::from_ptr(ff::avcodec_get_name((*self.enc).codec_id)).to_string_lossy().into_owned() }
    }

    /// Pick a subtitle codec the container supports and build decoder + encoder.
    /// `Ok(None)` when the stream is not text (bitmap subtitles cannot be converted).
    unsafe fn for_container(
        ofmt: *const ff::AVOutputFormat,
        ipar: *mut ff::AVCodecParameters,
        in_tb: AVRational,
    ) -> Result<Option<SubTranscoder>> {
        let src_id = (*ipar).codec_id;
        if !is_text_sub(src_id) {
            return Ok(None);
        }
        let mut cands = vec![ff::AV_CODEC_ID_SUBRIP, ff::AV_CODEC_ID_MOV_TEXT, ff::AV_CODEC_ID_WEBVTT, ff::AV_CODEC_ID_ASS];
        if src_id == ff::AV_CODEC_ID_ASS || src_id == ff::AV_CODEC_ID_SSA {
            cands.insert(0, ff::AV_CODEC_ID_ASS);
        }
        let mut target = None;
        for c in cands {
            if c != src_id
                && ff::avformat_query_codec(ofmt, c, ff::FF_COMPLIANCE_NORMAL as i32) == 1
                && !ff::avcodec_find_encoder(c).is_null()
            {
                target = Some(c);
                break;
            }
        }
        let Some(target) = target else { return Ok(None) };
        let dcodec = ff::avcodec_find_decoder(src_id);
        if dcodec.is_null() {
            return Err(IoError::Other("no subtitle decoder in this FFmpeg build".into()));
        }
        let ecodec = ff::avcodec_find_encoder(target);
        let mut dec = ff::avcodec_alloc_context3(dcodec);
        let mut enc = ff::avcodec_alloc_context3(ecodec);
        if dec.is_null() || enc.is_null() {
            ff::avcodec_free_context(&mut dec);
            ff::avcodec_free_context(&mut enc);
            return Err(IoError::Other("avcodec_alloc_context3 failed".into()));
        }
        let setup = || -> Result<()> {
            ffcheck(ff::avcodec_parameters_to_context(dec, ipar), "avcodec_parameters_to_context(sub)")?;
            (*dec).pkt_timebase = in_tb;
            ffcheck(ff::avcodec_open2(dec, dcodec, ptr::null_mut()), "open subtitle decoder")?;
            (*enc).time_base = SUB_TB;
            if !(*dec).subtitle_header.is_null() && (*dec).subtitle_header_size > 0 {
                let n = (*dec).subtitle_header_size as usize;
                let h = ff::av_malloc(n + 1) as *mut u8;
                ptr::copy_nonoverlapping((*dec).subtitle_header, h, n);
                *h.add(n) = 0;
                (*enc).subtitle_header = h;
                (*enc).subtitle_header_size = n as c_int;
            }
            ffcheck(ff::avcodec_open2(enc, ecodec, ptr::null_mut()), "open subtitle encoder")?;
            Ok(())
        };
        match setup() {
            Ok(()) => Ok(Some(SubTranscoder { dec, enc, buf: vec![0u8; 1 << 20] })),
            Err(e) => {
                ff::avcodec_free_context(&mut dec);
                ff::avcodec_free_context(&mut enc);
                Err(e)
            }
        }
    }

    /// Decode one subtitle packet and return the re-encoded packets (timestamps in `SUB_TB`).
    fn transcode(&mut self, pkt: *mut ff::AVPacket, in_tb: AVRational) -> Result<Vec<Packet>> {
        unsafe {
            let mut sub: ff::AVSubtitle = std::mem::zeroed();
            let mut got: c_int = 0;
            let ret = ff::avcodec_decode_subtitle2(self.dec, &mut sub, &mut got, pkt);
            if ret < 0 {
                tracing::warn!("subtitle packet skipped: {}", av_strerror(ret));
                return Ok(Vec::new());
            }
            if got == 0 {
                return Ok(Vec::new());
            }
            let mut out = Vec::new();
            if sub.pts == NOPTS && (*pkt).pts != NOPTS {
                sub.pts = ff::av_rescale_q((*pkt).pts, in_tb, TB_US);
            }
            let start = sub.start_display_time as i64;
            let mut end = sub.end_display_time as i64;
            if end <= start && (*pkt).duration > 0 {
                end = start + ff::av_rescale_q((*pkt).duration, in_tb, SUB_TB);
            }
            sub.pts += start * 1000;
            sub.end_display_time = (end - start).max(0) as u32;
            sub.start_display_time = 0;
            let n = ff::avcodec_encode_subtitle(self.enc, self.buf.as_mut_ptr(), self.buf.len() as c_int, &mut sub);
            if n > 0 {
                let np = Packet::alloc()?;
                ffcheck(ff::av_new_packet(np.as_ptr(), n), "av_new_packet")?;
                ptr::copy_nonoverlapping(self.buf.as_ptr(), (*np.as_ptr()).data, n as usize);
                let pts = ff::av_rescale_q(sub.pts, TB_US, SUB_TB);
                (*np.as_ptr()).pts = pts;
                (*np.as_ptr()).dts = pts;
                (*np.as_ptr()).duration = sub.end_display_time as i64;
                out.push(np);
            } else if n < 0 {
                tracing::warn!("subtitle encode failed: {}", av_strerror(n));
            }
            ff::avsubtitle_free(&mut sub);
            Ok(out)
        }
    }
}

impl Drop for SubTranscoder {
    fn drop(&mut self) {
        unsafe {
            ff::avcodec_free_context(&mut self.dec);
            ff::avcodec_free_context(&mut self.enc);
        }
    }
}

pub struct Output {
    oc: *mut ff::AVFormatContext,
    video_idx: i32,
    /// input stream index -> output stream index (-1 = not copied)
    map: Vec<i32>,
    in_tb: Vec<AVRational>,
    /// subtitle streams that are re-encoded (text -> a format the container accepts)
    subs: Vec<Option<SubTranscoder>>,
    enc_tb: AVRational,
    tmp_path: PathBuf,
    final_path: PathBuf,
    header_written: bool,
    finished: bool,
    pub copied_streams: usize,
    pub skipped: Vec<String>,
}

unsafe impl Send for Output {}

pub fn tmp_path_for(final_path: &Path) -> PathBuf {
    let stem = final_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "out".into());
    let ext = final_path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    final_path.with_file_name(format!("{stem}.fbtmp{ext}"))
}

fn is_mov_family(ext: &str) -> bool {
    matches!(ext.to_ascii_lowercase().as_str(), "mp4" | "m4v" | "mov" | "3gp")
}

impl Output {
    /// Create the muxer, add the video stream (from `enc`) and every audio / subtitle stream
    /// of `src`, copy chapters + metadata, open the file and write the header.
    pub fn create(final_path: &Path, src: &InputCtx, vinfo: &VideoInfo, enc: &VideoEncoder) -> Result<Output> {
        if let Some(dir) = final_path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| IoError::Other(format!("cannot create output directory '{}': {e}", dir.display())))?;
            }
        }
        let tmp_path = tmp_path_for(final_path);
        let cfinal = cstring(&final_path.to_string_lossy())?;
        let ctmp = cstring(&tmp_path.to_string_lossy())?;
        let ext = final_path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        unsafe {
            let mut oc: *mut ff::AVFormatContext = ptr::null_mut();
            let ret = ff::avformat_alloc_output_context2(&mut oc, ptr::null(), ptr::null(), cfinal.as_ptr());
            if ret < 0 || oc.is_null() {
                return Err(IoError::Unsupported(format!(
                    "cannot pick an output container for '{}' (use .mp4, .mkv or .mov): {}",
                    final_path.display(),
                    av_strerror(ret)
                )));
            }
            // The muxer re-opens `url` for +faststart: it must name the real (temporary) file.
            ff::av_free((*oc).url as *mut std::ffi::c_void);
            (*oc).url = ff::av_strdup(ctmp.as_ptr());
            let mut out = Output {
                oc,
                video_idx: -1,
                map: vec![-1; src.nb_streams()],
                in_tb: vec![AVRational { num: 1, den: 1 }; src.nb_streams()],
                subs: (0..src.nb_streams()).map(|_| None).collect(),
                enc_tb: enc.time_base,
                tmp_path,
                final_path: final_path.to_path_buf(),
                header_written: false,
                finished: false,
                copied_streams: 0,
                skipped: Vec::new(),
            };

            // ---- video stream ----
            let vst = ff::avformat_new_stream(oc, ptr::null());
            if vst.is_null() {
                return Err(IoError::Other("avformat_new_stream failed".into()));
            }
            out.video_idx = (*vst).index;
            let vpar = (*vst).codecpar;
            ffcheck(ff::avcodec_parameters_from_context(vpar, enc.codec_ctx()), "avcodec_parameters_from_context")?;
            (*vst).time_base = enc.time_base;
            (*vst).avg_frame_rate = enc.cfg.fps;
            (*vst).sample_aspect_ratio = enc.cfg.sar;
            if is_mov_family(&ext) && enc.cfg.codec == VideoCodec::Hevc {
                (*vpar).codec_tag = 0x3163_7668; // 'hvc1' (plays in Apple software too)
            } else {
                (*vpar).codec_tag = 0;
            }
            let in_vst = src.stream(vinfo.stream_index);
            ff::av_dict_copy(&mut (*vst).metadata, (*in_vst).metadata, 0);
            (*vst).disposition = (*in_vst).disposition & !(ff::AV_DISPOSITION_ATTACHED_PIC as i32);
            if let Some(m) = vinfo.display_matrix {
                let sd = ff::av_packet_side_data_new(
                    &mut (*vpar).coded_side_data,
                    &mut (*vpar).nb_coded_side_data,
                    ff::AV_PKT_DATA_DISPLAYMATRIX,
                    36,
                    0,
                );
                if !sd.is_null() {
                    ptr::copy_nonoverlapping(m.as_ptr(), (*sd).data, 36);
                }
            }

            // ---- copied streams ----
            for i in 0..src.nb_streams() {
                if i == vinfo.stream_index {
                    continue;
                }
                let ist = src.stream(i);
                let ipar = (*ist).codecpar;
                let t = (*ipar).codec_type;
                if t != ff::AVMEDIA_TYPE_AUDIO && t != ff::AVMEDIA_TYPE_SUBTITLE {
                    continue;
                }
                let supported = ff::avformat_query_codec((*oc).oformat, (*ipar).codec_id, ff::FF_COMPLIANCE_NORMAL as i32);
                let name = std::ffi::CStr::from_ptr(ff::avcodec_get_name((*ipar).codec_id)).to_string_lossy().into_owned();
                let mut transcoder: Option<SubTranscoder> = None;
                if supported == 0 {
                    if t == ff::AVMEDIA_TYPE_SUBTITLE {
                        match SubTranscoder::for_container((*oc).oformat, ipar, (*ist).time_base) {
                            Ok(Some(tc)) => {
                                tracing::info!("subtitle stream #{i} ({name}) is converted to {}", tc.target_name());
                                transcoder = Some(tc);
                            }
                            Ok(None) => {}
                            Err(e) => tracing::warn!("subtitle stream #{i} ({name}): conversion unavailable: {e}"),
                        }
                    }
                    if transcoder.is_none() {
                        let msg = format!(
                            "stream #{i} ({name}) cannot be stored in .{ext} files and was dropped (try an .mkv output)"
                        );
                        tracing::warn!("{msg}");
                        out.skipped.push(msg);
                        continue;
                    }
                }
                let ost = ff::avformat_new_stream(oc, ptr::null());
                if ost.is_null() {
                    return Err(IoError::Other("avformat_new_stream failed".into()));
                }
                if let Some(tc) = &transcoder {
                    ffcheck(ff::avcodec_parameters_from_context((*ost).codecpar, tc.enc), "avcodec_parameters_from_context(sub)")?;
                    (*ost).time_base = SUB_TB;
                } else {
                    ffcheck(ff::avcodec_parameters_copy((*ost).codecpar, ipar), "avcodec_parameters_copy")?;
                    (*(*ost).codecpar).codec_tag = 0;
                    (*ost).time_base = (*ist).time_base;
                }
                (*ost).disposition = (*ist).disposition;
                ff::av_dict_copy(&mut (*ost).metadata, (*ist).metadata, 0);
                out.map[i] = (*ost).index;
                out.in_tb[i] = (*ist).time_base;
                out.subs[i] = transcoder;
                out.copied_streams += 1;
            }

            // ---- global metadata + chapters ----
            ff::av_dict_copy(&mut (*oc).metadata, (*src.as_ptr()).metadata, 0);
            for c in 0..(*src.as_ptr()).nb_chapters as usize {
                let ich = *(*src.as_ptr()).chapters.add(c);
                let och = ff::av_mallocz(std::mem::size_of::<ff::AVChapter>()) as *mut ff::AVChapter;
                if och.is_null() {
                    break;
                }
                (*och).id = (*ich).id;
                (*och).time_base = (*ich).time_base;
                (*och).start = (*ich).start;
                (*och).end = (*ich).end;
                ff::av_dict_copy(&mut (*och).metadata, (*ich).metadata, 0);
                ff::av_dynarray_add(
                    &mut (*oc).chapters as *mut _ as *mut std::ffi::c_void,
                    &mut (*oc).nb_chapters as *mut u32 as *mut c_int,
                    och as *mut std::ffi::c_void,
                );
            }

            // ---- open + header ----
            if ((*(*oc).oformat).flags & ff::AVFMT_NOFILE as i32) == 0 {
                let ret = ff::avio_open(&mut (*oc).pb, ctmp.as_ptr(), ff::AVIO_FLAG_WRITE as i32);
                if ret < 0 {
                    return Err(ff_err(&format!("cannot create '{}'", out.tmp_path.display()), ret));
                }
            }
            let mut opts: *mut ff::AVDictionary = ptr::null_mut();
            if is_mov_family(&ext) {
                let (k, v) = (CString::new("movflags").unwrap(), CString::new("+faststart").unwrap());
                ff::av_dict_set(&mut opts, k.as_ptr(), v.as_ptr(), 0);
            }
            let ret = ff::avformat_write_header(oc, &mut opts);
            ff::av_dict_free(&mut opts);
            if ret < 0 {
                return Err(ff_err("avformat_write_header", ret));
            }
            out.header_written = true;
            out.enc_tb = enc.time_base;
            Ok(out)
        }
    }

    pub fn video_stream_tb(&self) -> AVRational {
        unsafe { (**(*self.oc).streams.add(self.video_idx as usize)).time_base }
    }

    /// Write an encoder packet (timestamps in the encoder time base).
    pub fn write_video(&mut self, pkt: &mut Packet) -> Result<()> {
        unsafe {
            let p = pkt.as_ptr();
            ff::av_packet_rescale_ts(p, self.enc_tb, self.video_stream_tb());
            (*p).stream_index = self.video_idx;
            ffcheck(ff::av_interleaved_write_frame(self.oc, p), "av_interleaved_write_frame(video)")?;
        }
        Ok(())
    }

    /// Write a stream-copied packet from the demuxer.
    pub fn write_side(&mut self, sp: &mut SidePacket) -> Result<()> {
        let oi = match self.map.get(sp.stream_index) {
            Some(&i) if i >= 0 => i,
            _ => return Ok(()),
        };
        if let Some(tc) = self.subs[sp.stream_index].as_mut() {
            let pkts = tc.transcode(sp.pkt.as_ptr(), self.in_tb[sp.stream_index])?;
            for np in pkts {
                unsafe {
                    let p = np.as_ptr();
                    let otb = (**(*self.oc).streams.add(oi as usize)).time_base;
                    ff::av_packet_rescale_ts(p, SUB_TB, otb);
                    (*p).stream_index = oi;
                    (*p).pos = -1;
                    ffcheck(ff::av_interleaved_write_frame(self.oc, p), "av_interleaved_write_frame(sub)")?;
                }
            }
            return Ok(());
        }
        unsafe {
            let p = sp.pkt.as_ptr();
            let otb = (**(*self.oc).streams.add(oi as usize)).time_base;
            ff::av_packet_rescale_ts(p, self.in_tb[sp.stream_index], otb);
            (*p).stream_index = oi;
            (*p).pos = -1;
            ffcheck(ff::av_interleaved_write_frame(self.oc, p), "av_interleaved_write_frame(copy)")?;
        }
        Ok(())
    }

    /// Write the trailer, close the file and move it to its final name.
    pub fn finish(mut self) -> Result<()> {
        unsafe {
            let ret = ff::av_write_trailer(self.oc);
            self.close_io();
            self.finished = true;
            ffcheck(ret, "av_write_trailer")?;
        }
        std::fs::rename(&self.tmp_path, &self.final_path).map_err(|e| {
            IoError::Other(format!(
                "cannot move '{}' to '{}': {e}",
                self.tmp_path.display(),
                self.final_path.display()
            ))
        })?;
        Ok(())
    }

    /// Discard the output (no trailer, temp file removed).
    pub fn abort(mut self) {
        self.cleanup();
    }

    unsafe fn close_io(&mut self) {
        if !self.oc.is_null() {
            if !(*self.oc).oformat.is_null() && ((*(*self.oc).oformat).flags & ff::AVFMT_NOFILE as i32) == 0 {
                ff::avio_closep(&mut (*self.oc).pb);
            }
        }
    }

    fn cleanup(&mut self) {
        unsafe {
            self.close_io();
        }
        let _ = std::fs::remove_file(&self.tmp_path);
        self.finished = true;
    }

    pub fn final_path(&self) -> &Path {
        &self.final_path
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        if !self.finished {
            self.cleanup();
        }
        unsafe {
            if !self.oc.is_null() {
                ff::avformat_free_context(self.oc);
                self.oc = ptr::null_mut();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmp_names() {
        let p = tmp_path_for(Path::new("out/clip_fb.mp4"));
        assert_eq!(p.file_name().unwrap().to_string_lossy(), "clip_fb.fbtmp.mp4");
        let p = tmp_path_for(Path::new("a.mkv"));
        assert_eq!(p.file_name().unwrap().to_string_lossy(), "a.fbtmp.mkv");
    }
}

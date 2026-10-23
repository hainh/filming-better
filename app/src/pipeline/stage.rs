//! Stage plumbing: the item flowing between stages, the `Stage` trait and the thread loop.

use super::pool::Tensor;
use super::Ctl;
use crate::native;
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;

/// One frame travelling through the pipeline. Cloning is cheap (shared tensor).
#[derive(Clone)]
pub struct Item {
    pub t: Arc<Tensor>,
    /// A scene change happened between the previous frame and this one.
    pub cut: bool,
}

/// A processing stage. `process`/`finish` must leave every kernel they enqueued *completed*
/// (i.e. sync their own stream) before returning, because the emitted tensors are handed to
/// another thread/stream immediately.
pub trait Stage: Send {
    fn name(&self) -> &'static str;
    fn process(&mut self, item: Item, out: &mut Vec<Item>) -> Result<()>;
    fn finish(&mut self, out: &mut Vec<Item>) -> Result<()>;
}

/// Run a stage on the calling thread until its input channel closes.
pub fn run_stage(mut st: Box<dyn Stage>, rx: &Receiver<Item>, tx: &Sender<Item>, ctl: &Ctl) -> Result<()> {
    native::init(ctl.device).context("binding CUDA device to worker thread")?;
    let mut out: Vec<Item> = Vec::with_capacity(8);
    loop {
        if ctl.aborted() {
            return Ok(());
        }
        match rx.recv() {
            Ok(item) => {
                st.process(item, &mut out).with_context(|| format!("{} stage", st.name()))?;
                for o in out.drain(..) {
                    if tx.send(o).is_err() {
                        return Ok(()); // downstream gone (error elsewhere)
                    }
                }
            }
            Err(_) => break,
        }
    }
    if ctl.aborted() {
        return Ok(());
    }
    st.finish(&mut out).with_context(|| format!("{} stage (flush)", st.name()))?;
    for o in out.drain(..) {
        if tx.send(o).is_err() {
            break;
        }
    }
    Ok(())
}

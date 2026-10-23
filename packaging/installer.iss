; =============================================================================
; filming-better - Inno Setup 6 script (LZMA2 solid, adds the install dir to the USER PATH)
;
; Build (after packaging\make_portable.ps1 has produced dist\filming-better-<ver>-win64\):
;   iscc /DAppVersion=1.0.0 packaging\installer.iss
;   iscc /DAppVersion=1.0.0 /DIncludeEngines=1 packaging\installer.iss    (embed models\*.engine too)
;
; Output: dist\filming-better-<ver>-setup.exe
;
; SIZE POLICY: docs/blueprint.md sets a 70 MB limit for the installer (phase 4, step 4) but < 200 MB
; for the whole product. The two only fit together when the installer does NOT embed the engines:
; the default build ships the program + manifest only (engines are fetched by the optional post-
; install step / `filming-better models pull`). The portable zip carries the default engines.
; CI (release.yml) checks the installer against -MaxInstallerMB 70 via make_portable.ps1.
;
; Silent install (winget/Scoop/CI):  filming-better-1.0.0-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
; Without PATH:                      ... /TASKS="!addtopath"
; Requires Inno Setup 6.3+ (x64compatible architecture identifier).
; =============================================================================

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\filming-better-" + AppVersion + "-win64"
#endif
#define AppName "filming-better"
#define AppExe  "filming-better.exe"
; Change OWNER once the repository exists
#define AppURL  "https://github.com/OWNER/filming-better"

[Setup]
; NOTE: AppId must stay constant across versions (upgrades). Do not regenerate.
AppId={{6F5B2C84-3A51-4E0B-9D5C-1D1C7B3A9F42}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=filming-better contributors
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=filming-better-{#AppVersion}-setup
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; per-user install (no admin): {autopf} -> %LOCALAPPDATA%\Programs, PATH -> HKCU\Environment
PrivilegesRequired=lowest
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ChangesEnvironment=yes
WizardStyle=modern
UninstallDisplayName={#AppName}
; SignTool=signtool  ; <- enable when a code-signing certificate / SignPath is available (see release.yml)

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add the installation folder to the user PATH (run `filming-better` from any terminal)"; GroupDescription: "Command line:"

[Files]
Source: "{#SourceDir}\{#AppExe}";             DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\fb_native.dll";         DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\nvinfer_lean_11.dll";   DestDir: "{app}"; Flags: ignoreversion
; FFmpeg (LGPL) + app-local MSVC runtime + any other dependency DLL gathered by make_portable.ps1
Source: "{#SourceDir}\*.dll";                 DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\LICENSE";               DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\THIRD_PARTY_NOTICES";   DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\THIRD_PARTY_RUST.html"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\BUILD_CONFIG.txt";      DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\README.md";             DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\VERSION.txt";           DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\models\manifest.json";  DestDir: "{app}\models"; Flags: ignoreversion
#ifdef IncludeEngines
Source: "{#SourceDir}\models\*.engine";       DestDir: "{app}\models"; Flags: ignoreversion nocompression
#endif

[Run]
; Optional: download the default engines after install (needs internet; ~tens of MB)
Filename: "{app}\{#AppExe}"; Parameters: "models pull span_x2 rife_v4"; \
  Description: "Download the default AI models now (span_x2, rife_v4)"; \
  Flags: postinstall skipifsilent unchecked shellexec
Filename: "{app}\{#AppExe}"; Parameters: "gpu-info"; \
  Description: "Check the GPU / driver (gpu-info)"; \
  Flags: postinstall skipifsilent unchecked shellexec

[Code]
const
  EnvKey = 'Environment';

function NormalizeDir(const S: string): string;
begin
  Result := Uppercase(S);
  while (Length(Result) > 0) and (Result[Length(Result)] = '\') do
    Delete(Result, Length(Result), 1);
end;

procedure EnvAddPath(const Dir: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths) then
    Paths := '';
  if Pos(';' + NormalizeDir(Dir) + ';', ';' + NormalizeDir(Paths) + ';') > 0 then
    exit;
  if (Paths <> '') and (Paths[Length(Paths)] <> ';') then
    Paths := Paths + ';';
  Paths := Paths + Dir;
  RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths);
end;

procedure EnvRemovePath(const Dir: string);
var
  Paths, Rest, Item, Out: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths) then
    exit;
  Rest := Paths;
  Out := '';
  while Rest <> '' do
  begin
    P := Pos(';', Rest);
    if P = 0 then
    begin
      Item := Rest;
      Rest := '';
    end
    else
    begin
      Item := Copy(Rest, 1, P - 1);
      Delete(Rest, 1, P);
    end;
    if (Item <> '') and (NormalizeDir(Item) <> NormalizeDir(Dir)) then
    begin
      if Out <> '' then
        Out := Out + ';';
      Out := Out + Item;
    end;
  end;
  if Out <> Paths then
    RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Out);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    EnvAddPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    EnvRemovePath(ExpandConstant('{app}'));
end;

; Mullenizer 1.0 — per-user Windows installer.
; Open in Inno Setup (6.7+ or 7.x) and select Build > Compile.
; Keep AppId unchanged for future upgrades.

#define AppName "Mullenizer"
#define AppVersion "1.0"
#define AppExeName "Mullenizer.exe"

[Setup]
AppId=Mullenizer.Desktop
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Mullenizer
AppPublisherURL=https://github.com/cjmullen11/Mullenizer
AppSupportURL=https://github.com/cjmullen11/Mullenizer
DefaultDirName={localappdata}\Programs\Mullenizer
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\release
OutputBaseFilename=MullenizerSetup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
; Install the entire tested PyInstaller one-folder distribution.
Source: "..\dist\Mullenizer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Mullenizer"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\Mullenizer"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch Mullenizer"; Flags: nowait postinstall skipifsilent

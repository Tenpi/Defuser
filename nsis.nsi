!include MUI2.nsh
!define APP_NAME "Defuzers"
!define APP_EXE "main.exe"
!define INSTALL_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\${APP_NAME}"
!define UNINSTALL_REGKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
!define MUI_ICON "assets\icons\favicon.ico"
!define MUI_UNICON "assets\icons\favicon.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "assets\images\nsis-installer.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "assets\images\nsis-installer.bmp"
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_AUTOCLOSE
!define MUI_UNFINISHPAGE_AUTOCLOSE

Name "${APP_NAME}"
OutFile "app\${APP_NAME}.exe"
InstallDir "$INSTDIR"
InstallDirRegKey HKLM "${INSTALL_DIR_REGKEY}" ""
Icon "assets\icons\favicon.ico"
UninstallIcon "assets\icons\favicon.ico"
RequestExecutionLevel user

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Function .onInit
    StrCpy $INSTDIR "$DOCUMENTS\${APP_NAME}"
FunctionEnd

Section "Install" SecInstall
    SetOutPath "$INSTDIR"
    File /r "build\defuzers\*"
    CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
    WriteRegStr HKLM "${UNINSTALL_REGKEY}" "" "$INSTDIR\uninstaller.exe"
    WriteUninstaller "$INSTDIR\uninstaller.exe"
SectionEnd

Section "Uninstall" SecUninstall
    RMDir /r "$INSTDIR"
    Delete "$DESKTOP\${APP_NAME}.lnk"
    DeleteRegKey HKLM "${UNINSTALL_REGKEY}" 
SectionEnd 

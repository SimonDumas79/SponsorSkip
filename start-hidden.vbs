' Starts the SponsorSkip backend with no console window. Used by the
' Startup-folder shortcut, so the backend is running whenever Simon is
' logged in. Stop it from Task Manager (node.exe) or: npm start in a terminal.
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = here
shell.Run "cmd /c node server\server.mjs >> """ & here & "\backend.log"" 2>&1", 0, False

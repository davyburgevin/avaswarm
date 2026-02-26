$p = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue
if ($p) {
    $pid = $p | Select-Object -ExpandProperty OwningProcess -First 1
    if ($pid) { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }
}
Start-Process -FilePath 'C:\Users\davy.burgevin\AgentsToolkitProjects\project-swarm\.venv\Scripts\python.exe' -ArgumentList '-u','-m','swarm.main','--web' -WindowStyle Hidden

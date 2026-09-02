@echo off
echo ====================================================
echo Avvio del Server Web locale per CityFlow in corso...
echo ====================================================
echo.
echo Il browser si aprira' automaticamente a breve.
echo (NON chiudere questa finestra finche' usi il simulatore)
echo.

:: Avvia il server in background usando WSL
start /B wsl bash -c "cd CityFlow/frontend && python3 -m http.server 8080"

:: Aspetta un paio di secondi per dare tempo al server di avviarsi
timeout /t 2 >nul

:: Apre il browser all'indirizzo del server
start http://localhost:8080/

@echo off
echo Importing all tournaments...

echo.
echo American Express...
python import_tournament.py dk-actual --file data/american_express_2026/dk_actual.csv --tournament "American Express" --year 2026 --date 2026-01-19
python import_tournament.py fanteam --ft2 data/american_express_2026/fanteam_2.csv --ft10 data/american_express_2026/fanteam_10.csv --tournament "American Express" --year 2026

echo.
echo Farmers Insurance Open...
python import_tournament.py dk-actual --file data/farmers_insurance_open_2026/dk_actual.csv --tournament "Farmers Insurance Open" --year 2026 --date 2026-02-02
python import_tournament.py fanteam --ft2 data/farmers_insurance_open_2026/fanteam_2.csv --ft10 data/farmers_insurance_open_2026/fanteam_10.csv --tournament "Farmers Insurance Open" --year 2026

echo.
echo Genesis Invitational...
python import_tournament.py dk-proj --file data/genesis_2026/dk_projected.csv --tournament "Genesis Invitational" --year 2026 --date 2026-02-16
python import_tournament.py fanteam --ft2 data/genesis_2026/fanteam_2.csv --ft10 data/genesis_2026/fanteam_10.csv --tournament "Genesis Invitational" --year 2026

echo.
echo Pebble Beach Pro-Am...
python import_tournament.py dk-proj --file data/pebble_beach_2026/dk_projected.csv --tournament "Pebble Beach Pro-Am" --year 2026 --date 2026-02-09
python import_tournament.py dk-actual --file data/pebble_beach_2026/dk_actual.csv --tournament "Pebble Beach Pro-Am" --year 2026 --date 2026-02-09
python import_tournament.py fanteam --ft2 data/pebble_beach_2026/fanteam_2.csv --ft10 data/pebble_beach_2026/fanteam_10.csv --tournament "Pebble Beach Pro-Am" --year 2026

echo.
echo Phoenix Open...
python import_tournament.py dk-proj --file data/phoenix_open_2026/dk_projected.csv --tournament "Phoenix Open" --year 2026 --date 2026-02-02
python import_tournament.py dk-actual --file data/phoenix_open_2026/dk_actual.csv --tournament "Phoenix Open" --year 2026 --date 2026-02-02
python import_tournament.py fanteam --ft2 data/phoenix_open_2026/fanteam_2.csv --ft10 data/phoenix_open_2026/fanteam_10.csv --tournament "Phoenix Open" --year 2026

echo.
echo All done! Running summary...
python analyze.py summary

pause
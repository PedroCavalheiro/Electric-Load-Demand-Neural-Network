@echo off

for /l %%i in (1,1,12) do (
    start "Task %%i" /high cmd /c python train.py %%i
)

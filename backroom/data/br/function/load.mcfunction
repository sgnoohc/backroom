gamerule doImmediateRespawn true
scoreboard objectives add deaths deathCount
scoreboard objectives add handled dummy
scoreboard objectives add dnf dummy
scoreboard objectives add back_to_survival trigger
tellraw @a {"text":"[Backroom] Loaded. Players go Spectator on death.","color":"yellow"}

# Teams for labeling status
team add alive
team add dead

# Make the DEAD label obvious
team modify dead color red
team modify dead suffix {"text":" [DEAD]","color":"red","bold":true}

# (Optional) keep 'alive' clean (no suffix)
team modify alive suffix ""

# --- mark this player as DNF and stop their timer ---
scoreboard objectives add dnf dummy
scoreboard objectives add runTime dummy

# Mark DNF and remove from the active runners set so ticks stop counting for them
scoreboard players set @s dnf 1
tag @s remove runner

# (Optional) clear their personal timer so it won't be shown anywhere
# scoreboard players reset @s runTime

# If NO runners remain, stop the global timer loop (only if you use schedule)
execute unless entity @a[tag=runner] run schedule clear br:timer_tick

title @s actionbar {"text":"You died — DNF","color":"red"}

gamemode spectator @s
tp 101 203 101
title @s actionbar {"text":"You died → Spectator mode","color":"gray"}

# Mark visually as DEAD in the sidebar
team leave @s
team join dead @s

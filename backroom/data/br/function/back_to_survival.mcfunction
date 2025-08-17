# Only switch if you're actually in Spectator
execute if entity @s[gamemode=spectator] run gamemode survival @s

# Small quality-of-life bits (optional)
effect clear @s
title @s actionbar {"text":"Back to Survival","color":"yellow"}

# Always reset the trigger so it can be used again later
scoreboard players reset @s back_to_survival

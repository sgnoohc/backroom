# Finish triggered by exit plate
scoreboard objectives add runTime dummy
tellraw @s [{"text":"Finished! Time: ","color":"yellow"},{"score":{"name":"*","objective":"runTime"}},{"text":" ticks (~","color":"gray"},{"text":"seconds = ticks/20","color":"gray"},{"text":")"}]
tag @a remove runner
scoreboard players reset @a runTime
kill @e[type=minecraft:warden]
kill @e[type=minecraft:item,nbt={Item:{id:"minecraft:sculk_catalyst"}}]
fill 100 200 100 199 207 199 air replace minecraft:sculk
fill 100 200 100 199 207 199 air replace minecraft:sculk_vein
fill 100 200 100 199 207 199 air replace minecraft:sculk_catalyst
fill 100 200 100 199 207 199 air replace minecraft:sculk_sensor
fill 100 200 100 199 207 199 air replace minecraft:sculk_shrieker
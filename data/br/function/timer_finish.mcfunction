# Finish one runner (called as that player)
tellraw @s [{"text":"Finished! Time: ","color":"yellow"},{"score":{"name":"*","objective":"runTime"}},{"text":" ticks (~","color":"gray"},{"text":"seconds = ticks/20","color":"gray"},{"text":")"}]
tag @s remove runner

team leave @s
team join alive @s

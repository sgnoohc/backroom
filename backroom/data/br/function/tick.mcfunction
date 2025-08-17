# enable the trigger for all players every tick (prevents the "you cannot trigger this objective yet" message)
scoreboard players enable @a back_to_survival

scoreboard players add @a[tag=runner] runTime 1

# flip NEW deaths to spectator
execute as @a if score @s deaths > @s handled run function br:on_death

# when the trigger is used, run the handler
execute as @a[scores={back_to_survival=1..}] run function br:back_to_survival

# sync handled to current total deaths so we only trigger once per death
execute as @a run scoreboard players operation @s handled = @s deaths

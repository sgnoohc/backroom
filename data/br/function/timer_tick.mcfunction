# Increment timer for all runners each tick
scoreboard objectives add runTime dummy
scoreboard players add @a[tag=runner] runTime 1
execute if entity @a[tag=runner] run schedule function br:timer_tick 1t replace
title @a[tag=runner] actionbar {"text":"Time: "}
title @a[tag=runner] actionbar [{"text":"Time: "},{"score":{"name":"*","objective":"runTime"}},{"text":" ticks"}]
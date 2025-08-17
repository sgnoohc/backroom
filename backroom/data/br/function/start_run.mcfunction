# Start a timed run: rebuild arena, TP players, tag as runners, summon Wardens, start ticking
function br:clear_area
function br:build_layout
kill @e[type=minecraft:warden]
scoreboard objectives add runTime dummy
scoreboard objectives modify runTime displayname {"text":"Run Time (ticks)"}
tag @a add runner
scoreboard players set @a runTime 0
fill 101 201 101 101 201 101 minecraft:air
fill 101 201 101 101 201 102 minecraft:air
fill 101 201 101 101 201 103 minecraft:air
fill 101 201 101 101 201 104 minecraft:air
fill 101 201 101 101 201 105 minecraft:air
fill 101 201 101 101 201 106 minecraft:air
fill 101 201 101 101 201 107 minecraft:air
fill 101 201 101 101 201 108 minecraft:air
fill 101 201 101 101 201 109 minecraft:air
fill 101 201 101 101 201 110 minecraft:air
fill 101 201 101 101 202 101 minecraft:air
fill 101 201 101 101 202 102 minecraft:air
fill 101 201 101 101 202 103 minecraft:air
fill 101 201 101 101 202 104 minecraft:air
fill 101 201 101 101 202 105 minecraft:air
fill 101 201 101 101 202 106 minecraft:air
fill 101 201 101 101 202 107 minecraft:air
fill 101 201 101 101 202 108 minecraft:air
fill 101 201 101 101 202 109 minecraft:air
fill 101 201 101 101 202 110 minecraft:air
fill 101 201 101 101 203 101 minecraft:air
fill 101 201 101 101 203 102 minecraft:air
fill 101 201 101 101 203 103 minecraft:air
fill 101 201 101 101 203 104 minecraft:air
fill 101 201 101 101 203 105 minecraft:air
fill 101 201 101 101 203 106 minecraft:air
fill 101 201 101 101 203 107 minecraft:air
fill 101 201 101 101 203 108 minecraft:air
fill 101 201 101 101 203 109 minecraft:air
fill 101 201 101 101 203 110 minecraft:air
fill 101 201 101 102 201 101 minecraft:air
fill 101 201 101 102 201 102 minecraft:air
fill 101 201 101 102 201 103 minecraft:air
fill 101 201 101 102 201 104 minecraft:air
fill 101 201 101 102 201 105 minecraft:air
fill 101 201 101 102 201 106 minecraft:air
fill 101 201 101 102 201 107 minecraft:air
fill 101 201 101 102 201 108 minecraft:air
fill 101 201 101 102 201 109 minecraft:air
fill 101 201 101 102 201 110 minecraft:air
fill 101 201 101 102 202 101 minecraft:air
fill 101 201 101 102 202 102 minecraft:air
fill 101 201 101 102 202 103 minecraft:air
fill 101 201 101 102 202 104 minecraft:air
fill 101 201 101 102 202 105 minecraft:air
fill 101 201 101 102 202 106 minecraft:air
fill 101 201 101 102 202 107 minecraft:air
fill 101 201 101 102 202 108 minecraft:air
fill 101 201 101 102 202 109 minecraft:air
fill 101 201 101 102 202 110 minecraft:air
fill 101 201 101 102 203 101 minecraft:air
fill 101 201 101 102 203 102 minecraft:air
fill 101 201 101 102 203 103 minecraft:air
fill 101 201 101 102 203 104 minecraft:air
fill 101 201 101 102 203 105 minecraft:air
fill 101 201 101 102 203 106 minecraft:air
fill 101 201 101 102 203 107 minecraft:air
fill 101 201 101 102 203 108 minecraft:air
fill 101 201 101 102 203 109 minecraft:air
fill 101 201 101 102 203 110 minecraft:air
fill 101 201 101 103 201 101 minecraft:air
fill 101 201 101 103 201 102 minecraft:air
fill 101 201 101 103 201 103 minecraft:air
fill 101 201 101 103 201 104 minecraft:air
fill 101 201 101 103 201 105 minecraft:air
fill 101 201 101 103 201 106 minecraft:air
fill 101 201 101 103 201 107 minecraft:air
fill 101 201 101 103 201 108 minecraft:air
fill 101 201 101 103 201 109 minecraft:air
fill 101 201 101 103 201 110 minecraft:air
fill 101 201 101 103 202 101 minecraft:air
fill 101 201 101 103 202 102 minecraft:air
fill 101 201 101 103 202 103 minecraft:air
fill 101 201 101 103 202 104 minecraft:air
fill 101 201 101 103 202 105 minecraft:air
fill 101 201 101 103 202 106 minecraft:air
fill 101 201 101 103 202 107 minecraft:air
fill 101 201 101 103 202 108 minecraft:air
fill 101 201 101 103 202 109 minecraft:air
fill 101 201 101 103 202 110 minecraft:air
fill 101 201 101 103 203 101 minecraft:air
fill 101 201 101 103 203 102 minecraft:air
fill 101 201 101 103 203 103 minecraft:air
fill 101 201 101 103 203 104 minecraft:air
fill 101 201 101 103 203 105 minecraft:air
fill 101 201 101 103 203 106 minecraft:air
fill 101 201 101 103 203 107 minecraft:air
fill 101 201 101 103 203 108 minecraft:air
fill 101 201 101 103 203 109 minecraft:air
fill 101 201 101 103 203 110 minecraft:air
fill 101 201 101 104 201 101 minecraft:air
fill 101 201 101 104 201 102 minecraft:air
fill 101 201 101 104 201 103 minecraft:air
fill 101 201 101 104 201 104 minecraft:air
fill 101 201 101 104 201 105 minecraft:air
fill 101 201 101 104 201 106 minecraft:air
fill 101 201 101 104 201 107 minecraft:air
fill 101 201 101 104 201 108 minecraft:air
fill 101 201 101 104 201 109 minecraft:air
fill 101 201 101 104 201 110 minecraft:air
fill 101 201 101 104 202 101 minecraft:air
fill 101 201 101 104 202 102 minecraft:air
fill 101 201 101 104 202 103 minecraft:air
fill 101 201 101 104 202 104 minecraft:air
fill 101 201 101 104 202 105 minecraft:air
fill 101 201 101 104 202 106 minecraft:air
fill 101 201 101 104 202 107 minecraft:air
fill 101 201 101 104 202 108 minecraft:air
fill 101 201 101 104 202 109 minecraft:air
fill 101 201 101 104 202 110 minecraft:air
fill 101 201 101 104 203 101 minecraft:air
fill 101 201 101 104 203 102 minecraft:air
fill 101 201 101 104 203 103 minecraft:air
fill 101 201 101 104 203 104 minecraft:air
fill 101 201 101 104 203 105 minecraft:air
fill 101 201 101 104 203 106 minecraft:air
fill 101 201 101 104 203 107 minecraft:air
fill 101 201 101 104 203 108 minecraft:air
fill 101 201 101 104 203 109 minecraft:air
fill 101 201 101 104 203 110 minecraft:air
fill 101 201 101 105 201 101 minecraft:air
fill 101 201 101 105 201 102 minecraft:air
fill 101 201 101 105 201 103 minecraft:air
fill 101 201 101 105 201 104 minecraft:air
fill 101 201 101 105 201 105 minecraft:air
fill 101 201 101 105 201 106 minecraft:air
fill 101 201 101 105 201 107 minecraft:air
fill 101 201 101 105 201 108 minecraft:air
fill 101 201 101 105 201 109 minecraft:air
fill 101 201 101 105 201 110 minecraft:air
fill 101 201 101 105 202 101 minecraft:air
fill 101 201 101 105 202 102 minecraft:air
fill 101 201 101 105 202 103 minecraft:air
fill 101 201 101 105 202 104 minecraft:air
fill 101 201 101 105 202 105 minecraft:air
fill 101 201 101 105 202 106 minecraft:air
fill 101 201 101 105 202 107 minecraft:air
fill 101 201 101 105 202 108 minecraft:air
fill 101 201 101 105 202 109 minecraft:air
fill 101 201 101 105 202 110 minecraft:air
fill 101 201 101 105 203 101 minecraft:air
fill 101 201 101 105 203 102 minecraft:air
fill 101 201 101 105 203 103 minecraft:air
fill 101 201 101 105 203 104 minecraft:air
fill 101 201 101 105 203 105 minecraft:air
fill 101 201 101 105 203 106 minecraft:air
fill 101 201 101 105 203 107 minecraft:air
fill 101 201 101 105 203 108 minecraft:air
fill 101 201 101 105 203 109 minecraft:air
fill 101 201 101 105 203 110 minecraft:air
fill 101 201 101 106 201 101 minecraft:air
fill 101 201 101 106 201 102 minecraft:air
fill 101 201 101 106 201 103 minecraft:air
fill 101 201 101 106 201 104 minecraft:air
fill 101 201 101 106 201 105 minecraft:air
fill 101 201 101 106 201 106 minecraft:air
fill 101 201 101 106 201 107 minecraft:air
fill 101 201 101 106 201 108 minecraft:air
fill 101 201 101 106 201 109 minecraft:air
fill 101 201 101 106 201 110 minecraft:air
fill 101 201 101 106 202 101 minecraft:air
fill 101 201 101 106 202 102 minecraft:air
fill 101 201 101 106 202 103 minecraft:air
fill 101 201 101 106 202 104 minecraft:air
fill 101 201 101 106 202 105 minecraft:air
fill 101 201 101 106 202 106 minecraft:air
fill 101 201 101 106 202 107 minecraft:air
fill 101 201 101 106 202 108 minecraft:air
fill 101 201 101 106 202 109 minecraft:air
fill 101 201 101 106 202 110 minecraft:air
fill 101 201 101 106 203 101 minecraft:air
fill 101 201 101 106 203 102 minecraft:air
fill 101 201 101 106 203 103 minecraft:air
fill 101 201 101 106 203 104 minecraft:air
fill 101 201 101 106 203 105 minecraft:air
fill 101 201 101 106 203 106 minecraft:air
fill 101 201 101 106 203 107 minecraft:air
fill 101 201 101 106 203 108 minecraft:air
fill 101 201 101 106 203 109 minecraft:air
fill 101 201 101 106 203 110 minecraft:air
fill 101 201 101 107 201 101 minecraft:air
fill 101 201 101 107 201 102 minecraft:air
fill 101 201 101 107 201 103 minecraft:air
fill 101 201 101 107 201 104 minecraft:air
fill 101 201 101 107 201 105 minecraft:air
fill 101 201 101 107 201 106 minecraft:air
fill 101 201 101 107 201 107 minecraft:air
fill 101 201 101 107 201 108 minecraft:air
fill 101 201 101 107 201 109 minecraft:air
fill 101 201 101 107 201 110 minecraft:air
fill 101 201 101 107 202 101 minecraft:air
fill 101 201 101 107 202 102 minecraft:air
fill 101 201 101 107 202 103 minecraft:air
fill 101 201 101 107 202 104 minecraft:air
fill 101 201 101 107 202 105 minecraft:air
fill 101 201 101 107 202 106 minecraft:air
fill 101 201 101 107 202 107 minecraft:air
fill 101 201 101 107 202 108 minecraft:air
fill 101 201 101 107 202 109 minecraft:air
fill 101 201 101 107 202 110 minecraft:air
fill 101 201 101 107 203 101 minecraft:air
fill 101 201 101 107 203 102 minecraft:air
fill 101 201 101 107 203 103 minecraft:air
fill 101 201 101 107 203 104 minecraft:air
fill 101 201 101 107 203 105 minecraft:air
fill 101 201 101 107 203 106 minecraft:air
fill 101 201 101 107 203 107 minecraft:air
fill 101 201 101 107 203 108 minecraft:air
fill 101 201 101 107 203 109 minecraft:air
fill 101 201 101 107 203 110 minecraft:air
fill 101 201 101 108 201 101 minecraft:air
fill 101 201 101 108 201 102 minecraft:air
fill 101 201 101 108 201 103 minecraft:air
fill 101 201 101 108 201 104 minecraft:air
fill 101 201 101 108 201 105 minecraft:air
fill 101 201 101 108 201 106 minecraft:air
fill 101 201 101 108 201 107 minecraft:air
fill 101 201 101 108 201 108 minecraft:air
fill 101 201 101 108 201 109 minecraft:air
fill 101 201 101 108 201 110 minecraft:air
fill 101 201 101 108 202 101 minecraft:air
fill 101 201 101 108 202 102 minecraft:air
fill 101 201 101 108 202 103 minecraft:air
fill 101 201 101 108 202 104 minecraft:air
fill 101 201 101 108 202 105 minecraft:air
fill 101 201 101 108 202 106 minecraft:air
fill 101 201 101 108 202 107 minecraft:air
fill 101 201 101 108 202 108 minecraft:air
fill 101 201 101 108 202 109 minecraft:air
fill 101 201 101 108 202 110 minecraft:air
fill 101 201 101 108 203 101 minecraft:air
fill 101 201 101 108 203 102 minecraft:air
fill 101 201 101 108 203 103 minecraft:air
fill 101 201 101 108 203 104 minecraft:air
fill 101 201 101 108 203 105 minecraft:air
fill 101 201 101 108 203 106 minecraft:air
fill 101 201 101 108 203 107 minecraft:air
fill 101 201 101 108 203 108 minecraft:air
fill 101 201 101 108 203 109 minecraft:air
fill 101 201 101 108 203 110 minecraft:air
fill 101 201 101 109 201 101 minecraft:air
fill 101 201 101 109 201 102 minecraft:air
fill 101 201 101 109 201 103 minecraft:air
fill 101 201 101 109 201 104 minecraft:air
fill 101 201 101 109 201 105 minecraft:air
fill 101 201 101 109 201 106 minecraft:air
fill 101 201 101 109 201 107 minecraft:air
fill 101 201 101 109 201 108 minecraft:air
fill 101 201 101 109 201 109 minecraft:air
fill 101 201 101 109 201 110 minecraft:air
fill 101 201 101 109 202 101 minecraft:air
fill 101 201 101 109 202 102 minecraft:air
fill 101 201 101 109 202 103 minecraft:air
fill 101 201 101 109 202 104 minecraft:air
fill 101 201 101 109 202 105 minecraft:air
fill 101 201 101 109 202 106 minecraft:air
fill 101 201 101 109 202 107 minecraft:air
fill 101 201 101 109 202 108 minecraft:air
fill 101 201 101 109 202 109 minecraft:air
fill 101 201 101 109 202 110 minecraft:air
fill 101 201 101 109 203 101 minecraft:air
fill 101 201 101 109 203 102 minecraft:air
fill 101 201 101 109 203 103 minecraft:air
fill 101 201 101 109 203 104 minecraft:air
fill 101 201 101 109 203 105 minecraft:air
fill 101 201 101 109 203 106 minecraft:air
fill 101 201 101 109 203 107 minecraft:air
fill 101 201 101 109 203 108 minecraft:air
fill 101 201 101 109 203 109 minecraft:air
fill 101 201 101 109 203 110 minecraft:air
fill 101 201 101 110 201 101 minecraft:air
fill 101 201 101 110 201 102 minecraft:air
fill 101 201 101 110 201 103 minecraft:air
fill 101 201 101 110 201 104 minecraft:air
fill 101 201 101 110 201 105 minecraft:air
fill 101 201 101 110 201 106 minecraft:air
fill 101 201 101 110 201 107 minecraft:air
fill 101 201 101 110 201 108 minecraft:air
fill 101 201 101 110 201 109 minecraft:air
fill 101 201 101 110 201 110 minecraft:air
fill 101 201 101 110 202 101 minecraft:air
fill 101 201 101 110 202 102 minecraft:air
fill 101 201 101 110 202 103 minecraft:air
fill 101 201 101 110 202 104 minecraft:air
fill 101 201 101 110 202 105 minecraft:air
fill 101 201 101 110 202 106 minecraft:air
fill 101 201 101 110 202 107 minecraft:air
fill 101 201 101 110 202 108 minecraft:air
fill 101 201 101 110 202 109 minecraft:air
fill 101 201 101 110 202 110 minecraft:air
fill 101 201 101 110 203 101 minecraft:air
fill 101 201 101 110 203 102 minecraft:air
fill 101 201 101 110 203 103 minecraft:air
fill 101 201 101 110 203 104 minecraft:air
fill 101 201 101 110 203 105 minecraft:air
fill 101 201 101 110 203 106 minecraft:air
fill 101 201 101 110 203 107 minecraft:air
fill 101 201 101 110 203 108 minecraft:air
fill 101 201 101 110 203 109 minecraft:air
fill 101 201 101 110 203 110 minecraft:air
kill @e[type=minecraft:item]
tp @a[tag=runner] 101 201 101
gamemode adventure @a[tag=runner]
# reset everyone’s visible score so only runners show
scoreboard players reset @a runTime
scoreboard players set @a[tag=runner] runTime 0
# show the timer on the sidebar (choose sort order)
scoreboard objectives setdisplay sidebar runTime
team leave @a[tag=runner]
setblock 176 201 179 minecraft:air
summon minecraft:warden 176 201 179
setblock 124 201 146 minecraft:air
summon minecraft:warden 124 201 146
setblock 147 201 140 minecraft:air
summon minecraft:warden 147 201 140
setblock 149 201 196 minecraft:air
summon minecraft:warden 149 201 196
setblock 101 201 107 minecraft:air
summon minecraft:warden 101 201 107
setblock 163 201 159 minecraft:air
summon minecraft:warden 163 201 159
setblock 191 201 196 minecraft:air
summon minecraft:warden 191 201 196
setblock 145 201 130 minecraft:air
summon minecraft:warden 145 201 130
setblock 152 201 180 minecraft:air
summon minecraft:warden 152 201 180
setblock 172 201 154 minecraft:air
summon minecraft:warden 172 201 154
schedule function br:timer_tick 1t replace

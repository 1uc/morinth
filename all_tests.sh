pytest

( ./shock_tube.py >/dev/null ) & shock_tube_pid=$!
( ./cloud_shock.py >/dev/null ) & cloud_shock_pid=$!
( ./burgers_sine.py >/dev/null ) & burgers_sine_pid=$!
( ./dam_break.py >/dev/null ) & dam_break_pid=$!
( ./sound_wave.py >/dev/null ) & sound_wave_pid=$!

all_pids=($shock_tube_pid \
          $cloud_shock_pid \
          $burgers_sine_pid \
          $dam_break_pid \
          $sound_wave_pid)

for pid in ${all_pids[@]}
do
  wait ${pid}
  if [[ $? != 0 ]]
  then
    exit -1
  fi
done

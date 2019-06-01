#ifndef _WALL_TIMER_H
#define _WALL_TIMER_H
#include <chrono>

class WallTimer {
  std::chrono::time_point<std::chrono::high_resolution_clock> start_walltime, end_walltime;
  bool timer_running;
public:
  WallTimer() {
    start_walltime = std::chrono::high_resolution_clock::now();
    end_walltime = start_walltime;
    timer_running = false;
  }

  void start_wallclock() {
    timer_running = true;
    start_walltime = std::chrono::high_resolution_clock::now();
  }

  void stop_wallclock() {
    end_walltime = std::chrono::high_resolution_clock::now();
    timer_running = false;
  }

  double get_walltime() {
    if (timer_running) {
      decltype(start_walltime) current_walltime = std::chrono::high_resolution_clock::now();
      return static_cast<double>(std::chrono::duration<double>(current_walltime-start_walltime).count());
    } else {
      return static_cast<double>(std::chrono::duration<double>(end_walltime-start_walltime).count());
    }
  }
};
#endif

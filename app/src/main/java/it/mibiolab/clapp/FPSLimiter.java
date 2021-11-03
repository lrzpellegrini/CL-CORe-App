package it.mibiolab.clapp;

public class FPSLimiter {

    private final double fps;
    private final long timeIntervalMs;
    private double lastTime;

    public FPSLimiter(double fps) {
        this.fps = fps;
        this.timeIntervalMs = Math.round(1000.0 / fps);
        this.lastTime = 0;
    }

    public boolean updateTime(long timeMs) {
        if((timeMs - this.lastTime) > this.timeIntervalMs) {
            this.lastTime = timeMs;
            return true;
        }

        return false;
    }
}

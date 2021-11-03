package it.mibiolab.clapp.utils;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class FPSCounter {

    private final DescriptiveStatistics elapsedMsAvg;
    private long lastTime;

    public FPSCounter(int window, long startTimeMs) {
        this.elapsedMsAvg =  new DescriptiveStatistics(window);
        this.lastTime = startTimeMs;
    }

    public void updateTime(long timeMs) {
        long elapsedMs = timeMs - this.lastTime;
        this.lastTime = timeMs;
        this.elapsedMsAvg.addValue(elapsedMs);

    }

    public double getFps() {
        return 1000.0 / elapsedMsAvg.getMean();
    }

    public void clear(long startTimeMs) {
        this.lastTime = startTimeMs;
        this.elapsedMsAvg.clear();
    }
}

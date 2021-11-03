package it.mibiolab.clapp.utils;

import java.util.NoSuchElementException;

import it.mibiolab.clapp.FPSLimiter;

public class FPSConstrainedDeque {
    private final FPSLimiter limiter;
    private int currentElements;
    private int bufferSize;
    private int maxSessionElements;
    private int sessionElements;

    public FPSConstrainedDeque(int bufferSize, double maxFps) {
        this.bufferSize = bufferSize;
        this.limiter = new FPSLimiter(maxFps);
        this.currentElements = 0;
    }

    public void startSession(int maxSessionElements) {
        this.currentElements = 0;
        this.sessionElements = 0;
        this.maxSessionElements = maxSessionElements;
    }

    public int getBufferSize() {
        if(this.bufferSize <= 0) {
            return Integer.MAX_VALUE;
        }

        return this.bufferSize;
    }

    public int getMaxSessionElements() {
        if(this.maxSessionElements <= 0) {
            return Integer.MAX_VALUE;
        }

        return this.maxSessionElements;
    }

    public boolean canAddNewElement(long timeMs) {
        return (!isSessionTerminated()) &&
                (currentElements < getBufferSize()) &&
                this.limiter.updateTime(timeMs);
    }

    public void addNewElement(boolean running) {
        if(isSessionTerminated()) {
            throw new IllegalStateException("Session is terminated");
        }

        if(running) {
            if(currentElements >= getBufferSize()) {
                throw new IllegalStateException("The buffer is full, can't add a new element");
            }

            currentElements++;
        }

        sessionElements++;
    }

    public void removeElement() throws NoSuchElementException {
        if(currentElements <= 0) {
            throw new IllegalStateException("Buffer is already empty");
        }

        currentElements--;
    }

    public boolean isSessionTerminated() {
        return sessionElements >= getMaxSessionElements();
    }

}

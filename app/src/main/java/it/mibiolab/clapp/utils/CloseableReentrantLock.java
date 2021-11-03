package it.mibiolab.clapp.utils;

import java.util.concurrent.locks.ReentrantLock;

/*
    https://stackoverflow.com/a/46248923
 */
public final class CloseableReentrantLock extends ReentrantLock {

    public CloseableReentrantLock() {
        super();
    }

    public CloseableReentrantLock(boolean fair) {
        super(fair);
    }

    private final ResourceLock unlocker = new ResourceLock() {
        @Override
        public void close() {
            CloseableReentrantLock.this.unlock();
        }
    };

    /**
     * @return an {@link AutoCloseable} once the lock has been acquired.
     */
    public ResourceLock lockAsResource() {
        lock();
        return unlocker;
    }
}
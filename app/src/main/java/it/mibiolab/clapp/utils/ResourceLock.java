package it.mibiolab.clapp.utils;

/*
    https://stackoverflow.com/a/46248923
 */
public interface ResourceLock extends AutoCloseable {

    /**
     * Unlocking doesn't throw any checked exception.
     */
    @Override
    void close();
}

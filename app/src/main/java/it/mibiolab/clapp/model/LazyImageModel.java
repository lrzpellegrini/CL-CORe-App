package it.mibiolab.clapp.model;

import android.media.Image;

import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;

public class LazyImageModel implements AndroidCameraImageModel {

    private WeakReference<Image> image;
    private volatile byte[] yPlane;
    private volatile byte[] uPlane;
    private volatile byte[] vPlane;
    private final int imgW;
    private final int imgH;
    private final int cropX;
    private final int cropY;
    private final int cropW;
    private final int cropH;
    private final int rotation;

    public LazyImageModel(Image image, int cropX, int cropY, int cropW, int cropH, int rotation) {
        this.image = new WeakReference<>(image);
        this.yPlane = null;
        this.uPlane = null;
        this.vPlane = null;
        this.imgW = image.getWidth();
        this.imgH = image.getHeight();
        this.cropX = cropX;
        this.cropY = cropY;
        this.cropW = cropW;
        this.cropH = cropH;
        this.rotation = rotation;
    }

    public synchronized byte[] getYPlane() {
        if(yPlane == null) {
            this.getYUVBuffers();
        }

        return yPlane;
    }

    public synchronized byte[] getUPlane() {
        if(uPlane == null) {
            this.getYUVBuffers();
        }

        return uPlane;
    }

    public synchronized byte[] getVPlane() {
        if(vPlane == null) {
            this.getYUVBuffers();
        }

        return vPlane;
    }

    public int getImgW() {
        return imgW;
    }

    public int getImgH() {
        return imgH;
    }

    public int getCropX() {
        return cropX;
    }

    public int getCropY() {
        return cropY;
    }

    public int getCropW() {
        return cropW;
    }

    public int getCropH() {
        return cropH;
    }

    public int getRotation() {
        return rotation;
    }

    public void setInUse() {
        this.getYUVBuffers();
    }

    protected synchronized void getYUVBuffers() {
        if(yPlane != null) {
            return;
        }

        Image img = image.get();
        if(img == null) {
            throw new IllegalStateException("The image is gone!");
        }

        ByteBuffer Ybuffer = img.getPlanes()[0].getBuffer();
        ByteBuffer Ubuffer = img.getPlanes()[1].getBuffer();
        ByteBuffer Vbuffer = img.getPlanes()[2].getBuffer();
        yPlane = new byte[Ybuffer.capacity()];
        uPlane = new byte[Ubuffer.capacity()];
        vPlane = new byte[Vbuffer.capacity()];

        Ybuffer.get(yPlane);
        Ubuffer.get(uPlane);
        Vbuffer.get(vPlane);

        image = null;
    }

    public static LazyImageModel fromImage(Image image, int cropX, int cropY,
                                           int cropW, int cropH, int rotation) {
        return new LazyImageModel(image, cropX, cropY, cropW, cropH, rotation);
    }
}

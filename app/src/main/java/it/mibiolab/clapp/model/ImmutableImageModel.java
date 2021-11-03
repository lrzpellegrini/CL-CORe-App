package it.mibiolab.clapp.model;

import android.media.Image;

import java.nio.ByteBuffer;

public final class ImmutableImageModel implements AndroidCameraImageModel {
    private final byte[] yPlane;
    private final byte[] uPlane;
    private final byte[] vPlane;
    private final int imgW;
    private final int imgH;
    private final int cropX;
    private final int cropY;
    private final int cropW;
    private final int cropH;
    private final int rotation;

    public ImmutableImageModel(byte[] yPlane, byte[] uPlane, byte[] vPlane,
                       int imgW, int imgH, int cropX, int cropY, int cropW, int cropH, int rotation) {
        this.yPlane = yPlane;
        this.uPlane = uPlane;
        this.vPlane = vPlane;
        this.imgW = imgW;
        this.imgH = imgH;
        this.cropX = cropX;
        this.cropY = cropY;
        this.cropW = cropW;
        this.cropH = cropH;
        this.rotation = rotation;
    }

    public byte[] getYPlane() {
        return yPlane;
    }

    public byte[] getUPlane() {
        return uPlane;
    }

    public byte[] getVPlane() {
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

    public static ImmutableImageModel fromImage(Image image, int cropX, int cropY,
                                                int cropW, int cropH, int rotation) {
        int w = image.getWidth();
        int h = image.getHeight();
        ByteBuffer Ybuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer Ubuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer Vbuffer = image.getPlanes()[2].getBuffer();
        byte[] Y = new byte[Ybuffer.capacity()];
        byte[] U = new byte[Ubuffer.capacity()];
        byte[] V = new byte[Vbuffer.capacity()];
        Ybuffer.get(Y);
        Ubuffer.get(U);
        Vbuffer.get(V);

        return new ImmutableImageModel(Y, U, V, w, h, cropX, cropY, cropW, cropH, rotation);
    }
}

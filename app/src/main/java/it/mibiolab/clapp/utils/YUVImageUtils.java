package it.mibiolab.clapp.utils;

import android.graphics.Rect;
import android.graphics.YuvImage;

import java.io.ByteArrayOutputStream;

import it.mibiolab.clapp.model.AndroidCameraImageModel;

public final class YUVImageUtils {

    private YUVImageUtils() { }

    public static byte[] getJpegBytesFromYUV(AndroidCameraImageModel data) {
        Rect rect = new Rect(data.getCropX(), data.getCropY(),
                data.getCropX() + data.getCropW(), data.getCropY() + data.getCropH());


        ByteArrayOutputStream stream = new ByteArrayOutputStream();

        byte[] nv21 = new byte[data.getYPlane().length + data.getUPlane().length + data.getVPlane().length];

        System.arraycopy(data.getYPlane(), 0, nv21, 0, data.getYPlane().length);
        System.arraycopy(data.getVPlane(), 0, nv21, data.getYPlane().length, data.getVPlane().length);
        System.arraycopy(data.getUPlane(), 0, nv21, data.getVPlane().length + data.getYPlane().length, data.getUPlane().length);

        YuvImage img = new YuvImage(nv21, 17, data.getImgW(), data.getImgH(), null);
        img.compressToJpeg(rect, 100, stream);
        return stream.toByteArray();
    }
}

package it.mibiolab.clapp.model;

public interface AndroidCameraImageModel {

    byte[] getYPlane();

    byte[] getUPlane();

    byte[] getVPlane();

    int getImgW();

    int getImgH();

    int getCropX();

    int getCropY();

    int getCropW();

    int getCropH();

    int getRotation();
}

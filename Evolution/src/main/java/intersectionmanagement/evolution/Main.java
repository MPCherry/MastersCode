package intersectionmanagement.evolution;

import com.amazonaws.AmazonServiceException;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import intersectionmanagement.simulator.Utility;
import java.util.Random;

import java.io.*;

public class Main {

    private static final int VERSION = 2;

    public static void main(String[] args) throws IOException {
        Random randomTagGen = new Random();
        String randomTag = String.format("%05d", randomTagGen.nextInt(100000));
        Evolution evolution = new Evolution(args[0], Utility.loadResource(args[1]), args[1]);
        evolution.runNE();
    }
}

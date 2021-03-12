package intersectionmanagement.trial;

import intersectionmanagement.simulator.Simulator;
import org.json.JSONObject;
import org.lwjgl.LWJGLException;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Logger;


public class Main {
    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) throws IOException {
        String parameters = new String(Files.readAllBytes(Paths.get(args[0])), StandardCharsets.UTF_8);
            Trial trial = new Trial(parameters);
        try {
            trial.runSimulationRendered();
        } catch (LWJGLException e) {
            e.printStackTrace();
        }
    }

    public static int evaluate_candidate(String parametersFp, String neural_network) throws IOException {
        String parametersStr = new String(Files.readAllBytes(Paths.get(parametersFp)), StandardCharsets.UTF_8);

        Random random = new Random();

        JSONObject nn = new JSONObject(neural_network);
        JSONObject parametersJSON = new JSONObject(parametersStr);
        parametersJSON.put("seed", random.nextInt(Integer.MAX_VALUE));
        parametersJSON.put("neural_network", nn.get("neural_network"));

        String parameters = parametersJSON.toString();

        Trial trial = new Trial(parameters);
        Simulator sim = trial.runSimulation();
        return sim.collisions/2;
    }
}

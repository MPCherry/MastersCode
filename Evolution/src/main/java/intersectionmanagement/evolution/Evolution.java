package intersectionmanagement.evolution;

import intersectionmanagement.simulator.Simulator;
import intersectionmanagement.simulator.Utility;
import intersectionmanagement.trial.Trial;
import org.apache.commons.lang3.SerializationUtils;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.ea.population.BasicPopulation;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.species.BasicSpecies;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.ml.genetic.crossover.Splice;
import org.encog.ml.genetic.genome.DoubleArrayGenome;
import org.encog.ml.genetic.genome.DoubleArrayGenomeFactory;
import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.neural.hyperneat.substrate.SubstrateNode;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.NEATUtil;
import org.encog.neural.neat.training.NEATGenome;
import org.encog.neural.neat.training.opp.NEATMutateAddLink;
import org.encog.neural.neat.training.opp.NEATMutateAddNode;
import org.encog.neural.neat.training.opp.NEATMutateRemoveLink;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class Evolution {
    private static final Logger LOGGER = Logger.getLogger(Evolution.class.getName());
    SimpleFormatter simpleFormatter;

    private TrainEA evolution;
    private int trialRepetitions;
    private int iterations;
    private List<JSONObject> trials;
    private Random random = new Random();
    private boolean neatTopology;


    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %5$s%6$s%n");
    }

    Evolution(String algorithm, String parameters, String experiment_name) throws IOException {
        String timeStamp = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date());

        LOGGER.info(String.format("Algorithm: %s\nEvolution parameters:\n%s", algorithm, parameters));
        JSONObject jsonParameters = new JSONObject(parameters);
        trialRepetitions = jsonParameters.getInt("trial_repetitions");
        iterations = jsonParameters.getInt("iterations");
        int populationSize = jsonParameters.getInt("population");

        trials = new ArrayList<>();
        JSONArray trialsJSON = jsonParameters.getJSONArray("trials");
        for (Object trialObject : trialsJSON) {
            String trial = Utility.loadResource((String) trialObject);
            trials.add(new JSONObject(trial));
        }

        Population population;
        CalculateScore score;
        switch (algorithm) {
            case "cne":
                neatTopology = false;
                population = getBasicPopulation(populationSize);
                score = new CNEScore();
                evolution = new TrainEA(population, score);
                evolution.addOperation(0.8, new Splice(10));
                evolution.addOperation(0.4, new MutatePerturbFixed(1.0));
                break;
            case "neat":
                neatTopology = true;
                population = getNEATPopulation(populationSize);
                score = new NEATScore();
                evolution = NEATUtil.constructNEATTrainer((NEATPopulation) population, score);
                evolution.addOperation(0.001d, new NEATMutateAddNode());
                evolution.addOperation(0.005d, new NEATMutateAddLink());
                evolution.addOperation(0.0005d, new NEATMutateRemoveLink());
                break;
            case "hyperneat":
                neatTopology = true;
                population = getHyperNEATPopulation(populationSize);
                score = new NEATScore();
                evolution = NEATUtil.constructNEATTrainer((NEATPopulation) population, score);
                break;
            default:
                LOGGER.severe(String.format("%s is not a valid evolution type", algorithm));
                throw new RuntimeException("No valid evolution algorithm in evolution parameters");
        }
    }

    public void runNE() throws IOException {
        LOGGER.info("Beginning evolution");
        long time = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            evolution.iteration();
            LOGGER.info(String.format("%03d %05d", i, (int) evolution.getBestGenome().getScore()));

            byte[] bestNetwork;
            if (neatTopology) {
                NEATGenome bestNNGenome = (NEATGenome) evolution.getBestGenome();
                NEATNetwork bestNN = (NEATNetwork) evolution.getCODEC().decode(bestNNGenome);
                bestNetwork = SerializationUtils.serialize(bestNN);
            } else {
                DoubleArrayGenome bestBasicGenome = (DoubleArrayGenome) evolution.getBestGenome();
                BasicNetwork bestBasicNetwork = setupSimpleNN(bestBasicGenome.getData());
                bestNetwork = SerializationUtils.serialize(bestBasicNetwork);
            }
            JSONObject bestNetworkJSON = new JSONObject();
            bestNetworkJSON.put("neural_network", bestNetwork);
            LOGGER.info("Serialized best network: " + bestNetworkJSON.toString());

            if (evolution.getBestGenome().getScore() == 0) {
                break;
            }
        }
        time = System.nanoTime() - time;
        LOGGER.info("Finishing evolution");
        LOGGER.info(String.format("Total time: %.1f", time/1000000000.f));
        evolution.finishTraining();
    }

    private static Population getBasicPopulation(int populationSize) {
        Population population = new BasicPopulation(populationSize, null);

        BasicSpecies defaultSpecies = new BasicSpecies();
        defaultSpecies.setPopulation(population);
        for (int i = 0; i < populationSize; i++) {
            final DoubleArrayGenome genome = generateBasicGenome();
            defaultSpecies.getMembers().add(genome);
        }
        population.setGenomeFactory(new DoubleArrayGenomeFactory(510));
        population.getSpecies().add(defaultSpecies);

        return population;
    }

    private static DoubleArrayGenome generateBasicGenome() {
        DoubleArrayGenome genome = new DoubleArrayGenome(510);
        final double[] organism = genome.getData();
        Random rng = new Random();
        for (int i = 0; i < organism.length; i++) {
            organism[i] = rng.nextDouble() * 2 - 1;
        }
        return genome;
    }

    private static NEATPopulation getNEATPopulation(int populationSize) {
        NEATPopulation population = new NEATPopulation(14, 2, populationSize);
        population.setInitialConnectionDensity(1.0);
        population.reset();
        return population;
    }

    private static NEATPopulation getHyperNEATPopulation(int populationSize) {
        Substrate substrate = constructCarSubstrate();
        NEATPopulation population = new NEATPopulation(substrate, populationSize);
        population.setInitialConnectionDensity(1.0);
        population.reset();
        return population;
    }

    private static Substrate substrate;

    private static Substrate constructCarSubstrate() {
        if (substrate != null) {
            return substrate;
        }

        substrate = new Substrate(2);

        SubstrateNode node0 = substrate.createInputNode();
        node0.getLocation()[0] = Math.cos(0) * 10;
        node0.getLocation()[1] = Math.sin(0) * 10;
        SubstrateNode node7 = substrate.createInputNode();
        node7.getLocation()[0] = Math.cos(Math.PI) * 10;
        node7.getLocation()[1] = Math.sin(Math.PI) * 10;
        SubstrateNode node1 = substrate.createInputNode();
        node1.getLocation()[0] = Math.cos(-0.3) * 10;
        node1.getLocation()[1] = Math.sin(-0.3) * 10;
        SubstrateNode node2 = substrate.createInputNode();
        node2.getLocation()[0] = Math.cos(0.3) * 10;
        node2.getLocation()[1] = Math.sin(0.3) * 10;
        SubstrateNode node3 = substrate.createInputNode();
        node3.getLocation()[0] = Math.cos(-0.65) * 10;
        node3.getLocation()[1] = Math.sin(-0.65) * 10;
        SubstrateNode node4 = substrate.createInputNode();
        node4.getLocation()[0] = Math.cos(0.65) * 10;
        node4.getLocation()[1] = Math.sin(0.65) * 10;
        SubstrateNode node5 = substrate.createInputNode();
        node5.getLocation()[0] = Math.cos(1.0) * 10;
        node5.getLocation()[1] = Math.sin(1.0) * 10;
        SubstrateNode node6 = substrate.createInputNode();
        node6.getLocation()[0] = Math.cos(-1.0) * 10;
        node6.getLocation()[1] = Math.sin(-1.0) * 10;
        SubstrateNode node8 = substrate.createInputNode();
        node8.getLocation()[0] = Math.cos(0.5*Math.PI) * 10;
        node8.getLocation()[1] = Math.sin(0.5*Math.PI) * 10;
        SubstrateNode node9 = substrate.createInputNode();
        node9.getLocation()[0] = Math.cos(-0.5*Math.PI) * 10;
        node9.getLocation()[1] = Math.sin(-0.5*Math.PI) * 10;
        SubstrateNode node10 = substrate.createInputNode();
        node10.getLocation()[0] = Math.cos(2) * 10;
        node10.getLocation()[1] = Math.sin(2) * 10;
        SubstrateNode node11 = substrate.createInputNode();
        node11.getLocation()[0] = Math.cos(-2) * 10;
        node11.getLocation()[1] = Math.sin(-2) * 10;
        SubstrateNode node12 = substrate.createInputNode();
        node12.getLocation()[0] = Math.cos(2.5) * 10;
        node12.getLocation()[1] = Math.sin(2.5) * 10;
        SubstrateNode node13 = substrate.createInputNode();
        node13.getLocation()[0] = Math.cos(-2.5) * 10;
        node13.getLocation()[1] = Math.sin(-2.5) * 10;

        // It's really important to add output nodes last, otherwise the links aren't setup correctly
        // Probably have to add hidden nodes after output nodes as well
        SubstrateNode speedNode;
        speedNode = substrate.createOutputNode();
        speedNode.getLocation()[0] = 0;
        speedNode.getLocation()[1] = -5;

        SubstrateNode turningNode;
        turningNode = substrate.createOutputNode();
        turningNode.getLocation()[0] = 0;
        turningNode.getLocation()[1] = 5;

        for (SubstrateNode inputNode : substrate.getInputNodes()) {
            substrate.createLink(inputNode, substrate.getOutputNodes().get(0));
            substrate.createLink(inputNode, substrate.getOutputNodes().get(1));
        }

        return substrate;
    }

    private int runTrials(byte[] neuralNetwork) {
        int totalCollisions = 0;

        for (JSONObject trial : trials) {
            trial.put("neural_network", neuralNetwork);
            for (int i = 0; i < trialRepetitions; i++) {
                trial.put("seed", random.nextInt(Integer.MAX_VALUE));
                Trial client = new Trial(trial.toString());
                Simulator sim = client.runSimulation();
                totalCollisions += sim.collisions/2;
            }
        }

        return totalCollisions;
    }

    private class CNEScore implements CalculateScore {
        @Override
        public double calculateScore(MLMethod phenotype) {
            DoubleArrayGenome genome = (DoubleArrayGenome) phenotype;
            BasicNetwork nn = setupSimpleNN(genome.getData());
            return runTrials(SerializationUtils.serialize(nn));
        }

        @Override
        public boolean shouldMinimize() {
            return true;
        }

        @Override
        public boolean requireSingleThreaded() {
            return false;
        }
    }

    private static BasicNetwork setupSimpleNN(double[] weights) {
        BasicNetwork neuralNetwork = new BasicNetwork();
        neuralNetwork.addLayer(new BasicLayer(null, true, 14));
        neuralNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), false, 30));
        neuralNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), false, 2));
        neuralNetwork.getStructure().finalizeStructure();

        int weightCounter = 0;
        for (int i = 0; i < 15; i++) {
            for (int j = 0; j < 30; j++) {
                neuralNetwork.setWeight(0, i, j, weights[weightCounter]);
                weightCounter++;
            }
        }

        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 2; j++) {
                neuralNetwork.setWeight(1, i, j, weights[weightCounter]);
                weightCounter++;
            }
        }

        return neuralNetwork;
    }

    private class NEATScore implements CalculateScore {
        @Override
        public double calculateScore(MLMethod phenotype) {
            NEATNetwork nn = (NEATNetwork) phenotype;
            return runTrials(SerializationUtils.serialize(nn));
        }

        @Override
        public boolean shouldMinimize() {
            return true;
        }

        @Override
        public boolean requireSingleThreaded() {
            return false;
        }
    }
}

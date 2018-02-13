package com.acc.app.geolearning4j.keras;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class MultiClasiffier {

	private static Logger log = LoggerFactory.getLogger(MultiClasiffier.class);

	public int predict()
			throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

		MultiLayerNetwork modelConfig = KerasModelImport.importKerasSequentialModelAndWeights("asdad", false);

		return 0;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}

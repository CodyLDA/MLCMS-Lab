package org.vadere.state.attributes.models;

import java.util.Arrays;
import java.util.List;

import org.vadere.annotation.factories.attributes.ModelAttributeClass;
import org.vadere.state.attributes.Attributes;

@ModelAttributeClass
public class AttributesSIRG extends Attributes {

	private int infectionsAtStart = 0;
	private double infectionRate = 0.01;
	private double infectionMaxDistance = 1;
	// Add recovery rate as a private attribute
	private double recoveryProbability = 0.4;

	public int getInfectionsAtStart() { return infectionsAtStart; }

	public double getInfectionRate() {
		return infectionRate;
	}

	public double getInfectionMaxDistance() {
		return infectionMaxDistance;
	}
        // Add getter for the recovery rate attribute
	public double getRecoveryProbability() {
		return recoveryProbability;
	}

}

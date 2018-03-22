from __future__ import print_function
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout
from parmed.amber import * 
import copy


pressure = 1*unit.atmospheres
temperature = 300*unit.kelvin
collision_rate = 10.0/unit.picoseconds
timestep = 2.0*unit.femtosecond 



base = AmberParm("SYSTEM.top","SYSTEM.crd") 
#create an openmm System from Amber parameters
system = base.createSystem(constraints=app.AllBonds,rigidWater=True,nonbondedMethod=app.CutoffPeriodic,nonbondedCutoff=1.0*unit.nanometers)


ref_sys = copy.deepcopy(system)
#take the non bonded forces
for force_index, reference_force in enumerate(ref_sys.getForces()):
    reference_force_name = reference_force.__class__.__name__ 
    if (reference_force_name == "NonbondedForce"):
        #now we have the nonbonded forces 
        test_force = reference_force
        #nonbonded_force = copy.deepcopy(reference_force)#copy the force 

ref_sys.addForce(test_force)


'''
barostat = mm.MonteCarloBarostat(pressure,temperature,25)
ref_sys.addForce(barostat)
#add a thermostat
thermostat = mm.AndersenThermostat(temperature,collision_rate)
ref_sys.addForce(thermostat)
#define the alchemical state
#alchemical_system = alchemical_region(reference_system,[0])
#create the alchemically modify nonbonded forces for the sodium ion
#alchemically_modified_nonbonded(reference_system,[0],switch_width=1.0*unit.nanometers)
#set the integrator
integrator = mm.VerletIntegrator(2.0*unit.femtoseconds)
'''
integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 
    2.0*unit.femtoseconds)
integrator.setConstraintTolerance(0.00001)

platform = mm.Platform.getPlatformByName('CUDA')
#properties = {'CudaPrecision': 'mixed'}
simulation = app.Simulation(base.topology, ref_sys, integrator, platform)#,properties)
simulation.context.setPositions(base.positions)

print('Minimizing...')
simulation.minimizeEnergy()

simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
print('Equilibrating...')
simulation.step(1000)

simulation.reporters.append(app.DCDReporter('trajectory.dcd', 1000))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, 
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=1000, separator='\t'))

print('Running Production...')
simulation.step(1000)
print('Done!')
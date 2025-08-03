package AxCore

import spinal.core._
import spinal.core.sim._
import spinal.sim.VCSFlags


object Config {

  private var SubDir: String = ""

  def setGenSubDir(subdir: String): Unit = {
    SubDir = subdir
  }

  def spinal = SpinalConfig(
    targetDirectory = "hw/gen/AxCore" + SubDir,
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = ASYNC,
      clockEdge = RISING,
      resetActiveLevel = LOW
    ),
    onlyStdLogicVectorAtTopLevelIo = true,    // what is this for?
    nameWhenByFile = false,                   // the generated Verilog codes will not have those "when_" wires
    anonymSignalPrefix = "tmp"                // use "tmp_" instead of "_zz_"
    // oneFilePerComponent = true,
  )

  // For Verilator + GTKWave Simulation
  def sim = SimConfig.withConfig(spinal).withFstWave

  // For VCS + Verdi Simulation
  def vcssim = SimConfig.withConfig(spinal).withVCS.withFSDBWave

}

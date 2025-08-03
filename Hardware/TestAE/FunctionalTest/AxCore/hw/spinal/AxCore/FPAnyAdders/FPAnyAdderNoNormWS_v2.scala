package AxCore.FPAnyAdders

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Basics.{FP2BinCvt, Bin2FPCvt}
import AxCore.Testing.TestCases.TU


// MARK: BlackBox for adder_fpany_no_norm_v2.v file
case class adder_fpany_no_norm_v2(Num: Int, ExpoWidth: Int, MantWidth: Int, Integer: Int, Fraction: Int) extends BlackBox {

  val TotalWidth = 1 + ExpoWidth + MantWidth
  val PWidth = Integer + Fraction

  // Generics
  addGeneric("NUM", Num)
  addGeneric("E", ExpoWidth)
  addGeneric("M", MantWidth)
  addGeneric("INT", Integer)
  addGeneric("FRAC", Fraction)

  val io = new Bundle {
    val psum   = in  Bits(ExpoWidth+PWidth+1 bits)
    val src    = in  Bits(Num*TotalWidth bits)
    val result = out Bits(ExpoWidth+PWidth+1 bits)    // {Sign, Exponent, Integer, Fraction}
  }
  noIoPrefix()

  // ? Be careful to the blackbox import path
  addRTLPath(s"hw/spinal/AxCore/BlackBoxImport/adder_fpany_no_norm_v2.v")

}


// MARK: FPAny Adder, Num of Src configurable, no Normalization, 2's complement, with SubNorm
case class FPAnyAdderNoNormWS_v2(Num: Int, ExpoWidth: Int, MantWidth: Int, Integer: Int, Fraction: Int) extends Component {

  val TotalWidth = 1 + ExpoWidth + MantWidth
  val PWidth = Integer + Fraction

  val io = new Bundle {
    val Src     = in  Vec(Bits(TotalWidth bits), Num)
    val PSumIn  = in  Bits(ExpoWidth+PWidth+1 bits)
    val PSumOut = out Bits(ExpoWidth+PWidth+1 bits)  simPublic()
  }
  noIoPrefix()

  // * Pack together
  val SrcPacked = Bits(Num*TotalWidth bits)
  SrcPacked := io.Src.asBits

  val AdderNoNorm = new adder_fpany_no_norm_v2(Num=Num, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction)
  AdderNoNorm.io.src := SrcPacked
  AdderNoNorm.io.psum := io.PSumIn
  io.PSumOut := AdderNoNorm.io.result

}



object FPAnyAdderNoNormWS_v2_RTL extends App {

  val Num = 8

  val ExpoWidth = 5
  val MantWidth = 10
  // val ExpoWidth = 8
  // val MantWidth = 7
  // val ExpoWidth = 8
  // val MantWidth = 23

  // val Integer  = 4
  val Integer  = 7
  val Fraction = MantWidth + 2

  Config.setGenSubDir(s"/NoNormAdder_v2/E${ExpoWidth}M${MantWidth}/N${Num}_Integer${Integer}")
  Config.spinal.generateVerilog(FPAnyAdderNoNormWS_v2(Num=Num, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction)).printRtl().mergeRTLSource()
}



object FPAnyAdderNoNormWS_v2_Sim extends App {

  val PERow = 2
  val PECol = 2
  val ExpoWidth = 5
  val MantWidth = 10

  val SimStartT = 10
  val PreloadLength = PERow
  val RunStartT = SimStartT + PreloadLength + 1
  val RunEndT = RunStartT + 8

  val AW1 = 10.25
  val AW2 = 2.5
  val AW1_FPBin = FP2BinCvt.FloatToFPAnyBin(f=AW1, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
  val AW2_FPBin = FP2BinCvt.FloatToFPAnyBin(f=AW2, ExpoWidth=ExpoWidth, MantWidth=MantWidth)

  Config.vcssim.compile{FPAnyAdderNoNormWS_v2(Num=2, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=4, Fraction=12)}.doSim { dut =>
    // simulation process
    dut.clockDomain.forkStimulus(2)
    // simulation code
    for (clk <- 0 until 100) {
      // test case
      if (clk < RunStartT || clk >= 90) {
        dut.io.Src.foreach{ r => r #= 0 }
        dut.io.PSumIn #= 0
      } else {
        dut.io.Src(0) #= BigInt(AW1_FPBin.replace("_", ""), 2)
        dut.io.Src(1) #= BigInt(AW2_FPBin.replace("_", ""), 2)
      }
      dut.clockDomain.waitRisingEdge()    // sample on rising edge
      val PSumOut = dut.io.PSumOut.toBigInt
      // val PSumOutBin = String.format("%22s", PSumOut.toString(2)).replace(' ', '0')
      // println(s"${PSumOutBin}")
      TU.NoNormPSumGetValue(PSumOut=PSumOut, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=ExpoWidth-1, Fraction=MantWidth+2)
    }
    sleep(50)
  }
}
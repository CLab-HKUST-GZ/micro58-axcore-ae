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
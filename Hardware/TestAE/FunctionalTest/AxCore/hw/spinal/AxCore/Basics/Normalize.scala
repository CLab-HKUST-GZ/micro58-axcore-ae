package AxCore.Basics

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps


case class normalize(ExpoWidth: Int, MantWidth: Int, Integer: Int, Fraction: Int) extends BlackBox {

  val TotalWidth = 1 + ExpoWidth + MantWidth
  val PWidth = Integer + Fraction

  // Generics
  addGeneric("E", ExpoWidth)
  addGeneric("M", MantWidth)
  addGeneric("INTEGER", Integer)
  addGeneric("FRACTION", Fraction)

  val io = new Bundle {
    val clk    = in  Bool()
    val src    = in  Bits(ExpoWidth+PWidth+1 bits)  simPublic()    // MARK: result={sign, expo, preserved_complement}
    val result = out Bits(TotalWidth bits)          simPublic()
  }
  noIoPrefix()

  // ? Be careful to the blackbox import path
  addRTLPath(s"hw/spinal/AxCore/BlackBoxImport/normalize.v")

  mapClockDomain(clock=io.clk)

}



case class NormalizeTop(ExpoWidth: Int, MantWidth: Int, Integer: Int, Fraction: Int) extends Component {

  val TotalWidth = 1 + ExpoWidth + MantWidth
  val PWidth = Integer + Fraction

  val io = new Bundle {
    val Src    = in  Bits(ExpoWidth+PWidth+1 bits)    // MARK: result={sign, expo, preserved_complement}
    val Result = out Bits(TotalWidth bits)
  }
  noIoPrefix()

  val Norm = new normalize(ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction)
  Norm.io.src := io.Src
  io.Result := Norm.io.result

}



object NormalizeTopRTL extends App {
  Config.setGenSubDir("/NormalizeTop")
  Config.spinal.generateVerilog(NormalizeTop(ExpoWidth=8, MantWidth=7, Integer=4, Fraction=15)).printRtl().mergeRTLSource()
}
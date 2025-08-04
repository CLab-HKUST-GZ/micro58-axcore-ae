package AxCore.FPAnyAdders

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Basics.{FP2BinCvt, Bin2FPCvt}


// MARK: BlackBox for fma_fpany.v file (with priority_encoder)
case class fma_fpany(MulExpoWidth: Int, MulMantWidth: Int, AddExpoWidth: Int, AddMantWidth: Int) extends BlackBox {

  val MulTotalWidth = 1 + MulExpoWidth + MulMantWidth
  val AddTotalWidth = 1 + AddExpoWidth + AddMantWidth

  // Generics
  addGeneric("E_MUL", MulExpoWidth)
  addGeneric("M_MUL", MulMantWidth)
  addGeneric("E_ADD", AddExpoWidth)
  addGeneric("M_ADD", AddMantWidth)

  val io = new Bundle {
    val clk      = in  Bool()
    val src0     = in  Bits(MulTotalWidth bits)
    val src1     = in  Bits(MulTotalWidth bits)
    val src2     = in  Bits(AddTotalWidth bits)
    val result   = out Bits(AddTotalWidth bits)    // result = src0 * src1 + src2
    val overflow = out Bool()
  }
  noIoPrefix()

  // ? Be careful to the blackbox import path
  addRTLPath(s"hw/spinal/AxCore/BlackBoxImport/fma_fpany.v")

  mapClockDomain(clock=io.clk)

}



// MARK: FPAny FuseMultAdd
case class FPAnyFMA(MulExpoWidth: Int, MulMantWidth: Int, AddExpoWidth: Int, AddMantWidth: Int) extends Component {

  val MulTotalWidth = 1 + MulExpoWidth + MulMantWidth
  val AddTotalWidth = 1 + AddExpoWidth + AddMantWidth

  val io = new Bundle {
    // val clk  = in  Bool()
    val Src0   = in  Bits(MulTotalWidth bits)
    val Src1   = in  Bits(MulTotalWidth bits)
    val Src2   = in  Bits(AddTotalWidth bits)
    val Result = out Bits(AddTotalWidth bits) simPublic()    // Result = Src0 * Src1 + Src2
    val Overflow = out Bool()
  }
  noIoPrefix()

  val FMA = new fma_fpany(MulExpoWidth=MulExpoWidth, MulMantWidth=MulMantWidth, AddExpoWidth=AddExpoWidth, AddMantWidth=AddMantWidth)
  FMA.io.src0 := io.Src0
  FMA.io.src1 := io.Src1
  FMA.io.src2 := io.Src2
  io.Result := FMA.io.result
  io.Overflow := FMA.io.overflow

}



object FPAnyFMA_RTL extends App {
  Config.setGenSubDir("/FPAnyFMA")
  Config.spinal.generateVerilog(FPAnyFMA(MulExpoWidth=5, MulMantWidth=10, AddExpoWidth=8, AddMantWidth=23)).printRtl().mergeRTLSource()
}
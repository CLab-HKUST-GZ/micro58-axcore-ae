package AxCore.SystolicArray_W8

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Operators.AdderInt
import AxCore.Basics.GuardAW
import AxCore.FPAnyAdders.FPAnyAdderNoNormWS_v2


// MARK: AxCore Two-Level Spatial Array, with Weight Stationary, with Shared Adder -- Tile
case class AxCore_SharedAdd_W8_Tile(
                                     PERow       : Int,
                                     PECol       : Int,
                                     QtExpoWidth : Int = 4,
                                     QtMantWidth : Int = 3,
                                     ExpoWidth   : Int,
                                     MantWidth   : Int,
                                     Integer     : Int,
                                     Fraction    : Int,
                                   ) extends Component {

  val TotalWidth   = 1 + ExpoWidth   + MantWidth
  val QtTotalWidth = 1 + QtExpoWidth + QtMantWidth
  val PWidth       = Integer + Fraction
  val PSumWidth    = ExpoWidth + PWidth + 1

  val io = new Bundle {
    val Wq_CIN_FP  = in  Vec(Bits(QtTotalWidth bits), PECol)         // Wq_FP Cascade In
    val Wq_COUT_FP = out Vec(Bits(QtTotalWidth bits), PECol)         // Wq_FP Cascade Out

    val T_CIN_TC   = in  Vec(Bits(TotalWidth bits), PERow)           // T_TC Cascade In
    val T_COUT_TC  = out Vec(Bits(TotalWidth bits), PERow)           // T_TC Cascade Out
    val A_Vld_CIN  = in  Vec(Bool(), PERow)                          // Validness of A_FP Cascade In
    val A_Vld_COUT = out Vec(Bool(), PERow)                          // Validness of A_FP Cascade Out

    val PSumIn    = in  Vec(Bits(PSumWidth bits), PECol)             // Pipelined PSum from Previous PE
    val PSumOut   = out Vec(Bits(PSumWidth bits), PECol)             // Updated PSum' to Next PE

    val WqLock     = in  Bool()                                      // Locking the preloaded Wq_FP
  }
  noIoPrefix()


  // * Generate PEs
  val PEs = List.tabulate(PERow,PECol)((pr, pc) => { AxCore_SharedAdd_W8_PE(
    QtExpoWidth=QtExpoWidth, QtMantWidth=QtMantWidth, ExpoWidth=ExpoWidth, MantWidth=MantWidth
  ).setName(s"PEs_pr${pr}_pc${pc}") })


  // * Shared Column Adders without Norm
  val SharedAddGroup = (0 until PECol).map{ pc =>
    new FPAnyAdderNoNormWS_v2(Num=PERow, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction).setName(s"SharedAdd_pc${pc}")
  }


  // * Horizontal
  for (pr <- 0 until PERow) {
    for (pc <- 0 until PECol) {
      if (pc == 0) {
        PEs(pr)(pc).io.T_CIN_TC  := io.T_CIN_TC(pr).setName(s"T_CIN_TC_pr${pr}")
        PEs(pr)(pc).io.A_Vld_CIN := io.A_Vld_CIN(pr).setName(s"A_Vld_CIN_pr${pr}")
      } else {
        PEs(pr)(pc).io.T_CIN_TC  := PEs(pr)(pc-1).io.T_COUT_TC
        PEs(pr)(pc).io.A_Vld_CIN := PEs(pr)(pc-1).io.A_Vld_COUT
      }
    }
    io.T_COUT_TC(pr).setName(s"M_COUT_TC_pr${pr}")   := PEs(pr)(PECol-1).io.T_COUT_TC
    io.A_Vld_COUT(pr).setName(s"A_Vld_COUT_pr${pr}") := PEs(pr)(PECol-1).io.A_Vld_COUT
  }


  // * Vertical
  for (pc <- 0 until PECol) {
    SharedAddGroup(pc).io.PSumIn := io.PSumIn(pc).setName(s"PSumIn_pc${pc}")
    io.PSumOut(pc).setName(s"PSumOut_pc${pc}") := SharedAddGroup(pc).io.PSumOut
    for (pr <- 0 until PERow) {
      SharedAddGroup(pc).io.Src(pr) := PEs(pr)(pc).io.R_FP
      if (pr == 0) {
        PEs(pr)(pc).io.Wq_CIN_FP := io.Wq_CIN_FP(pc).setName(s"Wq_CIN_FP_pc${pc}")
      } else {
        PEs(pr)(pc).io.Wq_CIN_FP := PEs(pr-1)(pc).io.Wq_COUT_FP
      }
    }
    io.Wq_COUT_FP(pc).setName(s"Wq_COUT_FP_c${pc}") := PEs(PERow-1)(pc).io.Wq_COUT_FP
  }


  // * One to all
  PEs.foreach{ pr => pr.foreach{ pc => pc.io.WqLock := io.WqLock }}

}



object AxCore_SharedAdd_W8_Tile_Gen extends App {
  // val ExpoWidth = 5
  // val MantWidth = 10
  val ExpoWidth = 8
  val MantWidth = 7
  val Integer   = ExpoWidth - 1
  val Fraction  = MantWidth + 2
  val Size      = 4
  Config.setGenSubDir(s"/W8_E${ExpoWidth}M${MantWidth}_Tile")
  Config.spinal.generateVerilog(AxCore_SharedAdd_W8_Tile(
    PERow=Size, PECol=Size, QtExpoWidth=4, QtMantWidth=3, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction
  )).printRtl().mergeRTLSource()
}



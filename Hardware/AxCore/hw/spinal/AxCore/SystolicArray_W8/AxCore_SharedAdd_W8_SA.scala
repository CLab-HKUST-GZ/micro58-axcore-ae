package AxCore.SystolicArray_W8

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Basics.{ParamsGen, PreAdd, normalize, AxDequant}


// MARK: AxCore Two-Level Spatial Array with Weight Stationary -- SA
case class AxCore_SharedAdd_W8_SA(
                                  TileRow     : Int,        // SA consists of (TileRow x TileCol) Tiles
                                  TileCol     : Int,        // SA consists of (TileRow x TileCol) Tiles
                                  PERow       : Int,        // Each Tile consists of (PERow x PECol) PEs
                                  PECol       : Int,        // Each Tile consists of (PERow x PECol) PEs
                                  QtExpoWidth : Int,
                                  QtMantWidth : Int,
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
    val DinTop_Wq_FP = in  Vec(Vec(Bits(QtTotalWidth bits), PECol), TileCol)
    val DinLft_A_FP  = in  Vec(Vec(Bits(TotalWidth bits), PERow), TileRow)
    val S_FP         = in  Bits(TotalWidth bits)
    val WqLock       = in  Bool()

    val Result       = out Vec(Vec(Bits(TotalWidth bits), PECol), TileCol)    // Pass Result Down to Accumulate
  }
  noIoPrefix()


  // * Params Generation
  val ParamsGenerate = new ParamsGen(ExpoWidth=ExpoWidth, MantWidth=MantWidth)
  ParamsGenerate.io.S_FP := io.S_FP


  // * PreAdd for A -> T
  val PreAddGroup = (0 until TileRow).map{ tr =>
    (0 until PERow).map{ pr =>
      new PreAdd(QtTotalWidth=QtTotalWidth, ExpoWidth=ExpoWidth, MantWidth=MantWidth).setName(s"PreAdd_tr${tr}_pr${pr}")
    }
  }

  for (tr <- 0 until TileRow) {
    for (pr <- 0 until PERow) {
      PreAddGroup(tr)(pr).io.A_FP := io.DinLft_A_FP(tr)(pr).setName(s"DinLft_A_FP_tr${tr}_pr${pr}")
    }
  }


  // * Generate tables of size TileRow*TileCol for DinLft Regs each is Vec(Regs, PERow)
  val DinLftRegTable_T = List.tabulate(TileRow,TileCol)((tr, tc) => {
    Vec((0 until PERow).map{ pr => Reg(Bits(TotalWidth bits)).init(B(0)).setName(s"DinLftRegTable_T_tr${tr}_tc${tc}_pr${pr}") })
  })    // for pipelining

  val DinLftRegTable_A_Vld = List.tabulate(TileRow,TileCol)((tr, tc) => {
    Vec((0 until PERow).map{ pr => Reg(Bool()).init(False).setName(s"DinLftRegTable_A_Vld_tr${tr}_tc${tc}_pr${pr}") })
  })    // for pipelining


  // * Generate a table of size (TileRow-1)*TileCol for Vertical PSum Regs each is Vec(Regs, PECol)
  val VertiRegTable_PSum = List.tabulate(TileRow-1,TileCol)((tr, tc) => {
    Vec((0 until PECol).map{ pc => Reg(Bits(PSumWidth bits)).init(B(0)).setName(s"VertiRegTable_PSum_tr${tr}_tc${tc}_pc${pc}") })
  })    // for pipelining


  // * Generate Tiles
  val Tiles = List.tabulate(TileRow,TileCol)((tr, tc) => { AxCore_SharedAdd_W8_Tile(    // MARK: SharedAdd
    PERow=PERow, PECol=PECol, QtExpoWidth=QtExpoWidth, QtMantWidth=QtMantWidth,
    ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction
  ).setName(s"Tiles_tr${tr}_tc${tc}") })


  // * Horizontal
  for (tr <- 0 until TileRow) {
    for (tc <- 0 until TileCol) {
      Tiles(tr)(tc).io.T_CIN_TC := DinLftRegTable_T(tr)(tc)         // Vec to Vec
      Tiles(tr)(tc).io.A_Vld_CIN := DinLftRegTable_A_Vld(tr)(tc)    // Vec to Vec
      if (tc == 0) {
        (0 until PERow).foreach{ pr => DinLftRegTable_T(tr)(0)(pr) := PreAddGroup(tr)(pr).io.T_TC }
        (0 until PERow).foreach{ pr => DinLftRegTable_A_Vld(tr)(0)(pr) := PreAddGroup(tr)(pr).io.A_Valid }
      } else {
        (0 until PERow).foreach{ pr => DinLftRegTable_T(tr)(tc)(pr) := Tiles(tr)(tc-1).io.T_COUT_TC(pr) }
        (0 until PERow).foreach{ pr => DinLftRegTable_A_Vld(tr)(tc)(pr) := Tiles(tr)(tc-1).io.A_Vld_COUT(pr) }
      }
    }
  }



  // * Vertical
  for (tc <- 0 until TileCol) {
    Tiles(0)(tc).io.Wq_CIN_FP := io.DinTop_Wq_FP(tc)
    Tiles(0)(tc).io.PSumIn := Vec(B(0), PECol)
    for (tr <- 0 until TileRow-1) {
      Tiles(tr+1)(tc).io.Wq_CIN_FP := Tiles(tr)(tc).io.Wq_COUT_FP
      VertiRegTable_PSum(tr)(tc) := Tiles(tr)(tc).io.PSumOut
      Tiles(tr+1)(tc).io.PSumIn := VertiRegTable_PSum(tr)(tc)
    }
  }


  // * One to all
  Tiles.foreach{ tr => tr.foreach{ tc => tc.io.WqLock := io.WqLock }}


  // * Normalizes
  val NormGroup = (0 until TileCol).map{ tc =>
    (0 until PECol).map{ pc =>
      new normalize(ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction).setName(s"Norm_tc${tc}_pc${pc}")
    }
  }


  // * Approx-Dequant
  val AxDequantGroup = (0 until TileCol).map{ tc =>
    (0 until PECol).map{ pc =>
      new AxDequant(ExpoWidth=ExpoWidth, MantWidth=MantWidth).setName(s"AxDequant_tc${tc}_pc${pc}")
    }
  }


  // * Postponing
  for (tc <- 0 until TileCol) {
    for (pc <- 0 until PECol) {
      // * Normalize the output PSum
      NormGroup(tc)(pc).io.src := Tiles(TileRow-1)(tc).io.PSumOut(pc)

      // * Dequantizing
      AxDequantGroup(tc)(pc).io.PSum_FP := NormGroup(tc)(pc).io.result
      AxDequantGroup(tc)(pc).io.S_minus_B2_plus_C2_TC := ParamsGenerate.io.S_minus_B2_plus_C2_TC

      // * Output of SA
      io.Result(tc)(pc).setName(s"Result_tc${tc}_pc${pc}") := AxDequantGroup(tc)(pc).io.Oact_FP
    }
  }


  // * Better naming
  (0 until TileCol).foreach(tc => (0 until PECol).foreach(pc => io.DinTop_Wq_FP(tc)(pc).setName(s"DinTop_Wq_FP_tc${tc}_pc${pc}")))

}



object AxCore_SharedAdd_W8_SA_Gen extends App {
  val TileNum = 16
  val PENum   = 4

  val ExpoWidth = 5
  val MantWidth = 10

  // val ExpoWidth = 8
  // val MantWidth = 7

  // val ExpoWidth = 8
  // val MantWidth = 23

  val Integer = ExpoWidth - 1
  val Fraction = MantWidth + 2

  Config.setGenSubDir(s"/Wq8_SA/AxCore_Wq8_E${ExpoWidth}M${MantWidth}_SA_T${TileNum}P${PENum}")
  Config.spinal.generateVerilog(AxCore_SharedAdd_W8_SA(
    TileRow=TileNum, TileCol=TileNum, PERow=PENum, PECol=PENum,
    QtExpoWidth=4, QtMantWidth=3, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction
  )).mergeRTLSource()  // .printRtl()
}
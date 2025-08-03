package AxCore.Testing.GoldenModels

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Basics.{FP2BinCvt, Bin2FPCvt}
import scala.math.pow
import AxCore.Testing.TestCases.TU


object mpFPMA_W4A16 {
  def calculate(
                 Wq_FP_In: BigInt, A_FP_In: BigInt,
                 ExpoWidth: Int, MantWidth: Int,
                 Wq_FmtSel: Int,
                 // StochasticBit: Boolean=false,
                 withComp: Boolean
               ): String = {

    // * Get the width according to Format Selection of Wq
    val QtExpoWidth = Wq_FmtSel match {
      case 2 => 2    // Wq_FmtSel=10 for E2M1
      case 3 => 1    // Wq_FmtSel=11 for E1M2
      case _ => 3    // Wq_FmtSel=00,01 for E3M0
    }

    val QtMantWidth = Wq_FmtSel match {
      case 2 => 1    // Wq_FmtSel=10 for E2M1
      case 3 => 2    // Wq_FmtSel=11 for E1M2
      case _ => 0    // Wq_FmtSel=00,01 for E3M0
    }

    // * -Bias for Wq
    val Minus_B1 = -(pow(2, QtExpoWidth-1)-1).toInt
    val Minus_B1_TC = Minus_B1.toBinaryString.takeRight(ExpoWidth)    // Two's Complement

    // * Constant Compensation
    val C1 = MantWidth match {
      case 10 => 43    // For A in E5M10, C1=43
      case 7  => 5     // For A in E8M7,  C1=5
      case _  => 0
    }
    val Comp = if (withComp) { C1 } else { 0 }

    // * -B1 + C1
    val C1_minus_B1 = (BigInt(Minus_B1_TC, 2) << MantWidth) + Comp

    // * T = A - B1 + C1
    val T = A_FP_In + C1_minus_B1
    val T_Bin = TU.BigIntToBinary(BI=T, Width=(1+ExpoWidth+MantWidth))

    // * StochasticBit
    val ChosenBit = s"${T_Bin(1+ExpoWidth)}"
    val StochasticBit = ChosenBit match {
      case "0" => false
      case "1" => true
    }
    // println(s"\nT_Bin=${T_Bin}, SB=${StochasticBit}")

    // * SNC
    val SNC_Out = GoldenModel_SNC_W4.calculate(Wq_FP_In=Wq_FP_In, Wq_FmtSel=Wq_FmtSel, StochasticBit=StochasticBit)
    val Wq_E3M2 = SNC_Out.Wq_FP_Out
    val Wq_NotZero = SNC_Out.Wq_NotZero

    // * Extracting
    val Wq_Sign     = (Wq_FP_In >> 3) & 1
    val Wq_ExpoMant = Wq_FP_In % 8                                        // Get the low 3 bits
    val Wq_E3M2_ExpoMant = Wq_E3M2 % 32                                  // Get the low 5 bits

    val A_Sign      = (A_FP_In >> (ExpoWidth + MantWidth)) & 1
    val A_ExpoMant  = A_FP_In % (BigInt(1) << (ExpoWidth + MantWidth))    // Get the low 15 bits

    // * Align Wq_E3M2
    val Wq_E3M2_Aligned = Wq_E3M2_ExpoMant << (MantWidth - 2)

    // * mpFPMA
    // val R_FP_Temp = Wq_E3M2_Aligned + A_FP_In + C1_minus_B1
    val R_FP_Temp = Wq_E3M2_Aligned + T
    val R_ExpoMant = R_FP_Temp % (BigInt(1) << (ExpoWidth + MantWidth))    // Get the low 15 bits

    // * Zero Check
    val isZero = (!Wq_NotZero) || (A_ExpoMant == 0)

    // * Output
    val ExpoMantWidth = ExpoWidth + MantWidth
    val R_ExpoMant_Bin = isZero match {
      case true  => String.format(s"%${ExpoMantWidth}s", BigInt(0).toString(2)).replace(' ', '0')
      case false => String.format(s"%${ExpoMantWidth}s", R_ExpoMant.toString(2)).replace(' ', '0')
    }
    val R_FPBin = s"${Wq_Sign ^ A_Sign}" + R_ExpoMant_Bin

    // * For Debug
    // println(s"Wq_FP_In_BigInt = ${Wq_FP_In}, Wq_Sign = ${Wq_Sign}, Wq_E3M2 = ${Wq_E3M2.toString(2)}, Wq_E3M2_ExpoMant = ${Wq_E3M2_ExpoMant.toString(2)}, Wq_E3M2_Aligned = ${Wq_E3M2_Aligned.toString(2)}")
    // println(s"A_Sign = ${A_Sign}, A_ExpoMant = ${A_ExpoMant.toString(2)}")
    // println(s"StochasticBit = ${StochasticBit}")
    // println(s"R_FPBin = ${R_FPBin}")
    // println(s"isZero = ${isZero}\n")

    R_FPBin
  }
}



object Test_mpFPMA {

  // Run test here, or run test at OverallFunctionalTest
  def main(args: Array[String]): Unit = {
    runTest()
  }

  def runTest(): Unit = {
    val Wq_FP = -1
    val QtExpoWidth = 2
    val QtMantWidth = 1
    val Wq_FmtSel = 2
    val A_FP  = 3.5
    val ExpoWidth = 5
    val MantWidth = 10
    val Wq_FPBin = FP2BinCvt.FloatToFPAnyBin(f=Wq_FP, ExpoWidth=QtExpoWidth, MantWidth=QtMantWidth, withNaNInf=false)
    val A_FPBin  = FP2BinCvt.FloatToFPAnyBin(f=A_FP, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
    val Wq_FPBigInt = BigInt(Wq_FPBin.replace("_", ""), 2)
    val A_FPBigInt  = BigInt(A_FPBin.replace("_", ""), 2)

    println(s"Wq = ${Wq_FP} = ${Wq_FPBin}")
    println(s"A = ${A_FP} = ${A_FPBin}")

    val R_FPBin = mpFPMA_W4A16.calculate(
      Wq_FP_In=Wq_FPBigInt, A_FP_In=A_FPBigInt,
      ExpoWidth=ExpoWidth, MantWidth=MantWidth,
      Wq_FmtSel=Wq_FmtSel,
      withComp=false
      // withComp=true
    )

    println(R_FPBin)
    println(Bin2FPCvt.FPAnyBinToFloat(R_FPBin, ExpoWidth=ExpoWidth, MantWidth=MantWidth))
  }

}
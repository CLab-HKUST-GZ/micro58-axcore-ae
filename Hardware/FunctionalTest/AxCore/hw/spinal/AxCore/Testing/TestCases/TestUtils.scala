package AxCore.Testing.TestCases
import AxCore.Basics.{FP2BinCvt, Bin2FPCvt}
import scala.math.pow
import scala.util.Random

// Test Utils
object TU {

  // * Convert BigInt (Binary Bits viewing as BigInt) into Binary Representation
  def BigIntToBinary(BI: BigInt, Width: Int): String = String.format(s"%${Width}s", BI.toString(2)).replace(' ', '0')

  // * Convert BigInt (Binary Bits viewing as BigInt) into Floating-Point Representation
  def BigIntToFP(
                  BI         : BigInt,
                  ExpoWidth  : Int,
                  MantWidth  : Int,
                  CustomBias : Option[Int] = None,
                  withNaNInf : Boolean = true
                ): Double = {

    val TotalWidth = 1 + ExpoWidth + MantWidth
    val FP_Bin = BigIntToBinary(BI, TotalWidth)
    val FP = Bin2FPCvt.FPAnyBinToFloat(FPBin=FP_Bin, ExpoWidth=ExpoWidth, MantWidth=MantWidth, CustomBias=CustomBias, WithNaNInf=withNaNInf)
    FP
  }

  // * Convert FP Double into corresponding binary representation in BigInt format for simulation usage
  def FPtoBinBigInt(
                    f          : Double,                  // FP value in double format
                    ExpoWidth  : Int,
                    MantWidth  : Int,
                    CustomBias : Option[Int] = None,
                    withNaNInf : Boolean = true
                  ): BigInt = {                           // Return FP's binary format
    BigInt(FP2BinCvt.FloatToFPAnyBin(f=f, ExpoWidth=ExpoWidth, MantWidth=MantWidth).replace("_", ""), 2)
  }

  def DynFP4toBinBigInt(
                         f      : Double,                 // FP value in double format
                         FmtSel : Int                     // Wq Format Select. 00,01 for E3M0, 10 for E2M1, 11 for E1M2
                       ): BigInt = {                      // Return FP's binary format
    FmtSel match {
      // Remember to turn off withNaNInf for FP4
      case 2 => BigInt(FP2BinCvt.FloatToFPAnyBin(f=f, ExpoWidth=2, MantWidth=1, withNaNInf=false).replace("_", ""), 2)
      case 3 => BigInt(FP2BinCvt.FloatToFPAnyBin(f=f, ExpoWidth=1, MantWidth=2, withNaNInf=false).replace("_", ""), 2)
      case _ => BigInt(FP2BinCvt.FloatToFPAnyBin(f=f, ExpoWidth=3, MantWidth=0, withNaNInf=false).replace("_", ""), 2)
    }
  }

  def FP4ValueSpaceAbs(FmtSel : Int) = {      // Wq Format Select. 00,01 for E3M0, 10 for E2M1, 11 for E1M2
    FmtSel match {
      case 2 => Array(0,       0.5, 1, 1.5, 2,      3,     4, 6       )    // E2M1
      case 3 => Array(0,       0.5, 1, 1.5, 2, 2.5, 3, 3.5            )    // E1M2
      case _ => Array(0, 0.25, 0.5, 1,      2,             4,    8, 16)    // E3M0
    }
  }

  def FP4ValueSpace(FmtSel : Int) = {      // Wq Format Select. 00,01 for E3M0, 10 for E2M1, 11 for E1M2
    FmtSel match {
      case 2 => Array(         -6, -4,       -3,       -2, -1.5, -1, -0.5       , 0,       0.5, 1, 1.5, 2,      3,     4, 6       )    // E2M1
      case 3 => Array(                 -3.5, -3, -2.5, -2, -1.5, -1, -0.5       , 0,       0.5, 1, 1.5, 2, 2.5, 3, 3.5            )    // E1M2
      case _ => Array(-16, -8,     -4,                 -2,       -1, -0.5, -0.25, 0, 0.25, 0.5, 1,      2,             4,    8, 16)    // E3M0
    }
  }

  def genRandomDynFP4Matrix(NumRows:Int, NumCols: Int, FmtSel: Int) = {
    val random = new Random()
    val Matrix = Array.ofDim[Double](NumRows, NumCols)
    val ChosenValueSpace = FP4ValueSpace(FmtSel)

    for (r <- 0 until NumRows) {
      for (c <- 0 until NumCols) {
        val randomIndex = random.nextInt(ChosenValueSpace.length)
        val randomElement = ChosenValueSpace(randomIndex)
        Matrix(r)(c) = randomElement
      }
    }
    Matrix
  }

  def NoNormPSumGetValue(PSumOut: BigInt, ExpoWidth: Int, MantWidth: Int, Integer: Int, Fraction: Int) = {
    val PSumOutBin = String.format("%22s", PSumOut.toString(2)).replace(' ', '0')
    val Sign = PSumOutBin(0)
    val Expo = PSumOutBin.slice(1, 6)
    val Inte = PSumOutBin.slice(6, 10)
    val Frac = PSumOutBin.slice(10, 22)
    val FixPointValue = BigInt(Inte, 2).toInt + (BigInt(Frac, 2).toInt * pow(2, -Fraction))
    val ExpoBias = pow(2, ExpoWidth-1) - 1
    val FinalValue = pow(2, BigInt(Expo, 2).toInt - ExpoBias) * FixPointValue
    // println(s"${PSumOutBin} => Sign=${Sign}, Expo=${Expo}, Inte=${Inte}, Frac=${Frac}, FinalValue=${FinalValue}")

    FinalValue
  }

}


// Generator : SpinalHDL v1.10.2a    git head : a348a60b7e8b6a455c72e1536ec3d74a2ea16935
// Component : AxCore_SharedAdd_W8_PE

`timescale 1ns/1ps

module AxCore_SharedAdd_W8_PE (
  input  wire [7:0]    Wq_CIN_FP,
  output wire [7:0]    Wq_COUT_FP,
  input  wire [31:0]   T_CIN_TC,
  output wire [31:0]   T_COUT_TC,
  input  wire          A_Vld_CIN,
  output wire          A_Vld_COUT,
  output wire [31:0]   R_FP,
  input  wire          WqLock,
  input  wire          clk,
  input  wire          resetn
);

  wire       [1:0]    SNC_Wq8_Wq_FmtSel;
  wire                SNC_Wq8_StochasticBit;
  wire                SNC_Wq8_Wq_NotZero;
  wire       [7:0]    SNC_Wq8_Wq_FP_Out;
  wire       [31:0]   AxMultS1_Sum;
  wire       [31:0]   GuardingAW_AW_Out;
  reg        [7:0]    WqLockReg;
  wire       [31:0]   Wq_Extend_FP;

  SNC_W8 SNC_Wq8 (
    .Wq_FP_In      (WqLockReg[7:0]        ), //i
    .Wq_FmtSel     (SNC_Wq8_Wq_FmtSel[1:0]), //i
    .StochasticBit (SNC_Wq8_StochasticBit ), //i
    .Wq_NotZero    (SNC_Wq8_Wq_NotZero    ), //o
    .Wq_FP_Out     (SNC_Wq8_Wq_FP_Out[7:0])  //o
  );
  AdderInt AxMultS1 (
    .X   (Wq_Extend_FP[31:0]), //i
    .Y   (T_CIN_TC[31:0]    ), //i
    .Sum (AxMultS1_Sum[31:0])  //o
  );
  GuardAW GuardingAW (
    .AW_In      (AxMultS1_Sum[31:0]     ), //i
    .Wq_NotZero (SNC_Wq8_Wq_NotZero     ), //i
    .A_Valid    (A_Vld_CIN              ), //i
    .AW_Out     (GuardingAW_AW_Out[31:0])  //o
  );
  assign SNC_Wq8_StochasticBit = T_CIN_TC[22];
  assign Wq_Extend_FP = {{{SNC_Wq8_Wq_FP_Out[7],4'b0000},SNC_Wq8_Wq_FP_Out[6 : 0]},20'h0};
  assign R_FP = GuardingAW_AW_Out;
  assign T_COUT_TC = T_CIN_TC;
  assign A_Vld_COUT = A_Vld_CIN;
  assign Wq_COUT_FP = WqLockReg;
  always @(posedge clk or negedge resetn) begin
    if(!resetn) begin
      WqLockReg <= 8'h0;
    end else begin
      if(WqLock) begin
        WqLockReg <= WqLockReg;
      end else begin
        WqLockReg <= Wq_CIN_FP;
      end
    end
  end


endmodule

module GuardAW (
  input  wire [31:0]   AW_In,
  input  wire          Wq_NotZero,
  input  wire          A_Valid,
  output wire [31:0]   AW_Out
);

  wire                SignTC;
  wire                Valid;

  assign SignTC = AW_In[31];
  assign Valid = (Wq_NotZero && A_Valid);
  assign AW_Out = (Valid ? AW_In : 32'h0);

endmodule

module AdderInt (
  input  wire [31:0]   X,
  input  wire [31:0]   Y,
  output wire [31:0]   Sum
);

  wire       [31:0]   SumSameWidth;

  assign SumSameWidth = (X + Y);
  assign Sum = SumSameWidth;

endmodule

module SNC_W8 (
  input  wire [7:0]    Wq_FP_In,
  input  wire [1:0]    Wq_FmtSel,
  input  wire          StochasticBit,
  output wire          Wq_NotZero,
  output reg  [7:0]    Wq_FP_Out
);

  wire                WqIsNorm;
  wire       [2:0]    WqMant;
  reg        [2:0]    SubNormCvt;
  wire                NeedRandomize;

  assign WqIsNorm = (|Wq_FP_In[6 : 3]);
  assign WqMant = Wq_FP_In[2 : 0];
  always @(*) begin
    case(WqMant)
      3'b101 : begin
        SubNormCvt = 3'b010;
      end
      3'b110 : begin
        SubNormCvt = 3'b100;
      end
      3'b111 : begin
        SubNormCvt = 3'b110;
      end
      default : begin
        SubNormCvt = 3'b000;
      end
    endcase
  end

  always @(*) begin
    if(WqIsNorm) begin
      Wq_FP_Out = Wq_FP_In;
    end else begin
      Wq_FP_Out = {Wq_FP_In[7 : 3],SubNormCvt};
    end
  end

  assign NeedRandomize = (Wq_FP_In[6 : 0] == 7'h02);
  assign Wq_NotZero = (NeedRandomize ? StochasticBit : (WqIsNorm || (|Wq_FP_In[2 : 1])));

endmodule

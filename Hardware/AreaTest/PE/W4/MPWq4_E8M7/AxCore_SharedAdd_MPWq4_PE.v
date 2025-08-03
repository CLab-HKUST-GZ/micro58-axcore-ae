// Generator : SpinalHDL v1.10.2a    git head : a348a60b7e8b6a455c72e1536ec3d74a2ea16935
// Component : AxCore_SharedAdd_MPWq4_PE

`timescale 1ns/1ps

module AxCore_SharedAdd_MPWq4_PE (
  input  wire [3:0]    Wq_CIN_FP,
  output wire [3:0]    Wq_COUT_FP,
  input  wire [1:0]    Wq_FmtSel,
  input  wire [15:0]   T_CIN_TC,
  output wire [15:0]   T_COUT_TC,
  input  wire          A_Vld_CIN,
  output wire          A_Vld_COUT,
  output wire [15:0]   R_FP,
  input  wire          WqLock,
  input  wire          clk,
  input  wire          resetn
);

  wire                SNC_MPWq4_StochasticBit;
  wire                SNC_MPWq4_Wq_NotZero;
  wire       [5:0]    SNC_MPWq4_Wq_FP_Out;
  wire       [15:0]   AxMultS1_Sum;
  wire       [15:0]   GuardingAW_AW_Out;
  reg        [3:0]    WqLockReg;
  wire       [15:0]   Wq_Extend_FP;

  SNC_W4 SNC_MPWq4 (
    .Wq_FP_In      (WqLockReg[3:0]          ), //i
    .Wq_FmtSel     (Wq_FmtSel[1:0]          ), //i
    .StochasticBit (SNC_MPWq4_StochasticBit ), //i
    .Wq_NotZero    (SNC_MPWq4_Wq_NotZero    ), //o
    .Wq_FP_Out     (SNC_MPWq4_Wq_FP_Out[5:0])  //o
  );
  AdderInt AxMultS1 (
    .X   (Wq_Extend_FP[15:0]), //i
    .Y   (T_CIN_TC[15:0]    ), //i
    .Sum (AxMultS1_Sum[15:0])  //o
  );
  GuardAW GuardingAW (
    .AW_In      (AxMultS1_Sum[15:0]     ), //i
    .Wq_NotZero (SNC_MPWq4_Wq_NotZero   ), //i
    .A_Valid    (A_Vld_CIN              ), //i
    .AW_Out     (GuardingAW_AW_Out[15:0])  //o
  );
  assign SNC_MPWq4_StochasticBit = T_CIN_TC[6];
  assign Wq_Extend_FP = {{{SNC_MPWq4_Wq_FP_Out[5],5'h0},SNC_MPWq4_Wq_FP_Out[4 : 0]},5'h0};
  assign R_FP = GuardingAW_AW_Out;
  assign T_COUT_TC = T_CIN_TC;
  assign A_Vld_COUT = A_Vld_CIN;
  assign Wq_COUT_FP = WqLockReg;
  always @(posedge clk or negedge resetn) begin
    if(!resetn) begin
      WqLockReg <= 4'b0000;
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
  input  wire [15:0]   AW_In,
  input  wire          Wq_NotZero,
  input  wire          A_Valid,
  output wire [15:0]   AW_Out
);

  wire                SignTC;
  wire                Valid;

  assign SignTC = AW_In[15];
  assign Valid = (Wq_NotZero && A_Valid);
  assign AW_Out = (Valid ? AW_In : 16'h0);

endmodule

module AdderInt (
  input  wire [15:0]   X,
  input  wire [15:0]   Y,
  output wire [15:0]   Sum
);

  wire       [15:0]   SumSameWidth;

  assign SumSameWidth = (X + Y);
  assign Sum = SumSameWidth;

endmodule

module SNC_W4 (
  input  wire [3:0]    Wq_FP_In,
  input  wire [1:0]    Wq_FmtSel,
  input  wire          StochasticBit,
  output wire          Wq_NotZero,
  output reg  [5:0]    Wq_FP_Out
);

  wire                Sign;
  wire       [2:0]    Content;
  wire       [4:0]    Decide;
  wire                NeedRandomize;

  assign Sign = Wq_FP_In[3];
  assign Content = Wq_FP_In[2 : 0];
  assign Decide = {Wq_FmtSel,Content};
  always @(*) begin
    case(Decide)
      5'h10 : begin
        Wq_FP_Out = {{{Sign,1'b0},3'b000},1'b0};
      end
      5'h11 : begin
        Wq_FP_Out = {{{Sign,1'b0},3'b000},1'b0};
      end
      5'h12 : begin
        Wq_FP_Out = {{{Sign,1'b0},Content},1'b0};
      end
      5'h13 : begin
        Wq_FP_Out = {{{Sign,1'b0},Content},1'b0};
      end
      5'h14 : begin
        Wq_FP_Out = {{{Sign,1'b0},Content},1'b0};
      end
      5'h15 : begin
        Wq_FP_Out = {{{Sign,1'b0},Content},1'b0};
      end
      5'h16 : begin
        Wq_FP_Out = {{{Sign,1'b0},Content},1'b0};
      end
      5'h17 : begin
        Wq_FP_Out = {{{Sign,1'b0},Content},1'b0};
      end
      5'h18 : begin
        Wq_FP_Out = {{Sign,2'b00},3'b000};
      end
      5'h19 : begin
        Wq_FP_Out = {{Sign,2'b00},3'b000};
      end
      5'h1a : begin
        Wq_FP_Out = {{Sign,2'b00},3'b000};
      end
      5'h1b : begin
        Wq_FP_Out = {{Sign,2'b00},3'b010};
      end
      5'h1c : begin
        Wq_FP_Out = {{Sign,2'b00},Content};
      end
      5'h1d : begin
        Wq_FP_Out = {{Sign,2'b00},Content};
      end
      5'h1e : begin
        Wq_FP_Out = {{Sign,2'b00},Content};
      end
      5'h1f : begin
        Wq_FP_Out = {{Sign,2'b00},Content};
      end
      default : begin
        Wq_FP_Out = {{Sign,Content},2'b00};
      end
    endcase
  end

  assign NeedRandomize = (Decide == 5'h19);
  assign Wq_NotZero = (NeedRandomize ? StochasticBit : (|Wq_FP_In[2 : 0]));

endmodule

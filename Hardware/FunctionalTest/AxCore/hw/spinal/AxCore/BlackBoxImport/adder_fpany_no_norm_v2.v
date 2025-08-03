module adder_fpany_no_norm_v2
#(
    parameter E = 5,
    parameter M = 10,
    parameter TOTAL = E + M + 1,
    parameter INT = 4,
    parameter FRAC = 12,
    parameter NUM = 4,
    parameter PWIDTH = INT + FRAC
)
(
    input   [E + PWIDTH     :0] psum    ,   //input at d0
    input   [NUM*TOTAL - 1  :0] src     ,   //input at d0
    output  [E + PWIDTH     :0] result      //output at d1
);
////////////////////////////////////////////////
wire                    src_sign[NUM - 1:0];
wire [E - 1         :0] src_exp [NUM - 1:0];
wire [M - 1         :0] src_man [NUM - 1:0];
wire                    psum_sign;
wire [E - 1         :0] psum_exp;
wire [PWIDTH - 1    :0] psum_man;

genvar i;
generate
    for (i = 0; i < NUM; i = i + 1) begin: unpack_src
        assign src_sign[i] = src[(i + 1)*(TOTAL) - 1];
        assign src_exp[i] = src[(i + 1)*(TOTAL) - 2 : i*(TOTAL) + M];
        assign src_man[i] = src[(i + 1)*(TOTAL) - E - 2 : i*(TOTAL)];
    end
endgenerate
assign psum_sign = psum[E + PWIDTH];
assign psum_exp  = psum[E + PWIDTH - 1:PWIDTH];
assign psum_man  = psum[PWIDTH - 1:0];
// ////////////////////////////////////////////////
// init data_exp
wire [E*(NUM + 1) - 1 :0] data_exp;

assign data_exp[E - 1:0] = psum_exp;

generate 
    for(i = 1; i < NUM + 1; i = i + 1) begin: init_exp_compare
        assign data_exp[E*(i + 1) - 1:E*i] = src_exp[i - 1];
    end
endgenerate
////////////////////////////////////////////////
// find max exp

wire [E - 1                 :0] exp_temp;

comp_tree #(
    .WIDTH(E),
    .NUM(NUM + 1)
)
u_comp_tree(
    .comp_data(data_exp),
    .max_data(exp_temp)
);
////////////////////////////////////////////////
// get diff number
wire [E - 1:0] exp_diff[NUM:0];    // Should be [NUM:0]

generate 
    for (i = 0; i < NUM + 1; i = i + 1) begin: diff
        assign exp_diff[i] = exp_temp - data_exp[E*(i + 1) - 1:E*i];
    end
endgenerate
////////////////////////////////////////////////
// transfer source code to complement
wire [PWIDTH :0] data_man_complement[NUM:0];

assign data_man_complement[0] = {psum_sign, psum_man};
generate 
    for (i = 1; i < NUM + 1; i = i + 1) begin: complement
        assign data_man_complement[i] = src_sign[i - 1] ? {{INT{1'b1}}, 1'b0, ~src_man[i - 1], {(FRAC-M){1'b1}}} + 'b1 : 
                                        {{INT{1'b0}}, 1'b1, src_man[i - 1], {(FRAC-M){1'b0}}};
    end
endgenerate
////////////////////////////////////////////////
// align
wire [PWIDTH :0] data_shifted[NUM:0];

generate
    for (i = 0; i < NUM + 1; i = i + 1) begin: align
        assign data_shifted[i] = $signed(data_man_complement[i]) >>> exp_diff[i];
    end
endgenerate
////////////////////////////////////////////////
// add
reg [PWIDTH :0] man_sum;

integer j;
always @(*) begin
    man_sum = 'b0;
    for (j = 0; j < NUM + 1; j = j + 1) begin: add
        man_sum = man_sum + data_shifted[j];
    end
end
////////////////////////////////////////////////
// complement output
wire                sign_d0;
wire [E - 1     :0] exp_d0;
wire [PWIDTH - 1:0] man_d0;

assign sign_d0 = man_sum[PWIDTH];
assign exp_d0 = exp_temp;
assign man_d0 = man_sum[PWIDTH - 1:0];
assign result = {sign_d0, exp_d0, man_d0};
////////////////////////////////////////////////
//dff #(PWIDTH+E+1) result_dff(.clock(clock), .d({sign_d0, exp_d0, man_d0} & {(PWIDTH+E+1){~clear}}), .q(result));
////////////////////////////////////////////////

endmodule





module comp_tree 
#(
    parameter WIDTH = 4,
    parameter NUM = 5
)
(
    input   [WIDTH*NUM - 1  :0] comp_data,
    output  [WIDTH - 1      :0] max_data
);

// power-of-two num
localparam NUM_LOW = 2 ** $clog2(NUM) / 2;
localparam NUM_HIGH = NUM - NUM_LOW;

generate
    if (NUM == 1) begin: comp_1
        assign max_data = comp_data;
    end 
    else if (NUM == 2) begin: comp_2
        wire compare;

        assign compare = comp_data[WIDTH - 1:0] < comp_data[WIDTH*2 - 1:WIDTH];
        assign max_data = compare ? comp_data[WIDTH*2 - 1:WIDTH] : comp_data[WIDTH - 1:0];  
    end 
    else if (NUM == 3) begin: comp_3
        wire                compare[1:0];
        wire [WIDTH - 1 :0] max_data_temp;
        
        assign compare[0] = comp_data[WIDTH - 1:0] < comp_data[WIDTH*2 - 1:WIDTH];
        assign max_data_temp = compare[0] ? comp_data[WIDTH*2 - 1:WIDTH] : comp_data[WIDTH - 1:0];

        assign compare[1] = max_data_temp < comp_data[WIDTH*3 - 1:WIDTH*2];
        assign max_data = compare[1] ? comp_data[WIDTH*3 - 1:WIDTH*2] : max_data_temp;
    end 
    else begin: comp_n

        wire [WIDTH - 1             :0] max_data_low, max_data_high;
        
        comp_tree #(
            .WIDTH(WIDTH),
            .NUM(NUM_LOW)
        )
        comp_tree_low (
            .comp_data(comp_data[WIDTH*NUM_LOW - 1:0]),
            .max_data(max_data_low)
        );

        comp_tree #(
            .WIDTH(WIDTH),
            .NUM(NUM_HIGH)
        )
        comp_tree_high (
            .comp_data(comp_data[WIDTH*NUM - 1:WIDTH*NUM_LOW]),
            .max_data(max_data_high)
        );

        wire compare;

        assign compare = max_data_low < max_data_high;
        assign max_data = compare ? max_data_high : max_data_low;  
    end
endgenerate

endmodule
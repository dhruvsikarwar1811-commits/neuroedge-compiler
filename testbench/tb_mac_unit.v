// NeuroEdge — MAC Unit Testbench
// Author: Yuvraj Sikarwar

module tb_mac_unit;
    reg        clk, rst_n, valid_in, acc_clear;
    reg  [7:0] activation, weight;
    wire [23:0] acc_out;
    wire        valid_out;

    mac_unit uut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in),
        .acc_clear(acc_clear),
        .activation(activation),
        .weight(weight),
        .acc_out(acc_out),
        .valid_out(valid_out)
    );

    always #5 clk = ~clk;

    initial begin
        clk=0; rst_n=0; valid_in=0; acc_clear=0;
        activation=0; weight=0;
        #20 rst_n=1;

        // Test 1: 5 * 3 = 15
        acc_clear=1; activation=8'd5; weight=8'd3; valid_in=1;
        #10 acc_clear=0;
        // Test 2: + 4*2 = 8, total=23
        activation=8'd4; weight=8'd2;
        #10 valid_in=0;
        #10;
        if(acc_out==24'd23)
            $display("TEST PASSED: acc_out=%d", acc_out);
        else
            $display("TEST FAILED: Expected 23, got %d", acc_out);

        $dumpfile("results/tb_mac_unit.vcd");
        $dumpvars(0, tb_mac_unit);
        #50 $finish;
    end
endmodule

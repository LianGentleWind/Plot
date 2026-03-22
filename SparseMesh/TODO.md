图3。64P和128P分两张图。数据需要修改，成本数据需要减去64*20000或者128*20000，再算成本/Sys Cost得的BW Cost，再归一化。不画其他数据了。
64P的SMesh default改名为SP，128P的SMesh default删除。64P新增FM-N16的配置，数值填成0即可。

\begin{table*}[!h]
\centering
\caption{Bandwidth and latency cost efficiency comparison across scales (64P / 128P)}
\label{tab:cost_efficiency_combined}
\begin{tabular}{l l c c c c c c}
\toprule
Scale & Topology & Latency ($\mu$s) & BW (Gb/s) & Sys Cost (\$) 
& BW Cost Eff. & Lat Cost Eff. & Net Cost \% \\
\midrule

\multirow{5}{*}{64P} 
& SparseMesh-N8 (default) & 22.16 & 100.0  & 1,379,840 & 0.67 & 1.00 & 7.24\% \\
& SparseMesh-N8 (opt)     & 22.16 & 149.98 & 1,379,840 & 1.00 & 1.00 & 7.24\% \\
& FullMesh-N8             & 21.99 & 200.03 & 1,387,520 & 1.33 & 1.00 & 7.75\% \\
& Torus-N8                & 22.24 & 100.0  & 1,379,840 & 0.67 & 1.00 & 7.24\% \\
& CLOS-N8                 & 23.10 & 171.43 & 1,470,920 & 1.07 & 0.90 & 12.98\% \\
\midrule

\multirow{5}{*}{128P}
& SparseMesh-N16 (default) & 22.19 & 40.0  & 2,775,040 & 0.73 & 1.00 & 7.75\% \\
& SparseMesh-N16 (SP)      & 22.19 & 55.05 & 2,775,040 & 1.00 & 1.00 & 7.75\% \\
& SparseMesh-N16 (opt)     & 22.17 & 58.52 & 2,775,040 & 1.06 & 1.00 & 7.75\% \\
& CLOS-N16                 & 23.14 & 80.0  & 2,941,840 & 1.37 & 0.91 & 12.98\% \\
& Torus-2D-N16             & 22.35 & 50.0  & 2,790,400 & 0.90 & 0.99 & 8.26\% \\
\bottomrule
\end{tabular}
\end{table*}  


图4。类似上一张图，画64P，128P，256P下的归一化Cost Efficiency。但是这里开始已经选取了Clos和SM各自的最优策略，所以没有那么多选型了，不再需要分子图，一张图里即可。使用竖向的柱状图。

\begin{table}[h]
\centering
\caption{Training throughput and cost efficiency comparison of 2048-NPU cluster}
\label{Training throughput and cost efficiency}
\begin{tabular}{lcccccc}
\toprule
Topology & \makecell{Norm.\\Throughput} & \makecell{Total\\Cost (\$)} & 
\makecell{Cost\\Efficiency} & \makecell{Net\\Cost \%} \\
\midrule
64P Clos & 1.000 & 47.8M & 1.000 & 14.31\% \\
64P SM & 0.933 & 43.4M & 1.027 & 5.65\% \\
\midrule
128P Clos & 1.009 & 47.8M & 1.009 & 14.31\% \\
128P SM & 0.932 & 43.4M & 1.027 & 5.65\% \\
\midrule
256P Clos & 1.014 & 47.8M & 1.014 & 14.31\% \\
256P SM & 0.933 & 43.4M & 1.028 & 5.65\% \\
\bottomrule
\end{tabular}
\end{table} 


图5。3种rack卡数，2种序列长度，3种TPOT一共18个自变量。绘制的还是Clos和SM各自的Thro吞吐值。
\begin{table}[h]
\centering
\caption{Inference performance comparison}
\label{tab:inference_detailed}
\begin{tabular}{ccccc c c}
\toprule
Scale & Seq Len & \makecell{TPOT\\(ms)} & \makecell{Clos\\Thro.} & \makecell{SM\\Thro.} & \makecell{Perf.\\Change} \\
\midrule

\multirow{6}{*}{64P}
& \multirow{3}{*}{4096}
& 20  & 64  & 64  & -0.16\% \\
&     & 50  & 204 & 202 & -0.78\% \\
&     & 100 & 417 & 413 & -0.90\% \\
\cmidrule(lr){2-6}
& \multirow{3}{*}{8192}
& 20  & 37  & 37  & +0.16\% \\
&     & 50  & 130 & 129 & -0.47\% \\
&     & 100 & 270 & 269 & -0.54\% \\

\midrule

\multirow{6}{*}{128P}
& \multirow{3}{*}{4096}
& 20  & 67  & 66  & -1.49\% \\
&     & 50  & 202 & 199 & -1.85\% \\
&     & 100 & 415 & 407 & -1.99\% \\
\cmidrule(lr){2-6}
& \multirow{3}{*}{8192}
& 20  & 42  & 41  & -1.21\% \\
&     & 50  & 129 & 128 & -1.15\% \\
&     & 100 & 269 & 266 & -1.26\% \\

\midrule

\multirow{6}{*}{256P}
& \multirow{3}{*}{4096}
& 20  & 66  & 56  & -14.87\% \\
&     & 50  & 202 & 173 & -14.16\% \\
&     & 100 & 414 & 355 & -14.23\% \\
\cmidrule(lr){2-6}
& \multirow{3}{*}{8192}
& 20  & 42  & 38  & -9.56\% \\
&     & 50  & 129 & 116 & -10.09\% \\
&     & 100 & 269 & 243 & -9.70\% \\

\bottomrule
\end{tabular}
\end{table}

图6
EP16
||吞吐||||DP暴露|||
|规模|256|512|1024||256|512|1024|
|SMesh 128P rack|1.72368|1.7199|1.69911||2.114|3.431|5.689|
|SMesh 64P rack|1.64997|1.64619|1.61217||2.544|4.117|6.761|
|Fullmesh|1.674|1.6668|1.62||2.253|3.813|6.788|
|CLOS|1|0.997|0.97||1.639|2.778|4.978|

EP32
||吞吐||||DP暴露|||
|规模|256|512|1024||256|512|1024|
|SMesh 128P rack|1.752079|1.748298|1.736958||3.9|5.75|8.897|
|SMesh 64P rack|1.701047|1.697267|1.685927||3.436|5.077|7.905|
|Fullmesh|1.682805|1.679205|1.666607||4.407|5.962|9.206|
|CLOS|1|0.998|0.991||3.294|4.519|6.76|

EP64
||吞吐||||DP暴露|||
|规模|256|512|1024||256|512|1024|
|SMesh 128P rack|1.81445|1.81256|1.80689||0.299|0.329|0.389| % 此行数据需脑补，下一行是脑补结果：
|SMesh 128P rack|1.86874|1.86247|1.85768||0.324|0.361|0.421|
|SMesh 64P rack|1.81445|1.81256|1.80689||0.299|0.329|0.389|
|Fullmesh|1.758396|1.756596|1.751197||0.284|0.313|0.37|
|CLOS|1|0.998|0.995||0.276|0.304|0.36|

吞吐数据：绘制柱状图。双栏图的上半部分，横向分为三个共享Y轴的小子图，分别对应3个EP分组。内部绘制不同规模256,512,1024下的吞吐性能，用不同颜色柱子表示几种拓扑。相同规模下的几种柱子应该紧密贴在一起。

DP暴露数据：绘制折线图，双栏图的下半部分。分为三个独立的小子图，分别绘制不同颜色表示的拓扑折线在规模变化时，DP暴露比例数值的趋势。


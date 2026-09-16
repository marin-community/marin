# Initial target math evaluation audit

Independent live audit passed. All eleven unique target receipts match the frozen specification, live permanent checkpoint metadata, exact scored-population hash, and plotted receipt. Target p0 is displayed once in each domain curve, producing twelve displayed target points from eleven measurements.

Maximum PALOMA token-loss discrepancy: 9.53674316e-07; tolerance: 5e-05.

| Curve | MATH-500 minimum / lowest observed | GSM8K minimum / lowest observed |
|---|---|---|
| wikipedia/matched | 30%, 4.768359 epochs, PPL 21.174635 | 20%, 3.161914 epochs, PPL 18.625457 |
| wikipedia/target | 50%, 7.937188 epochs, PPL 7.832023 | 20%, 3.160974 epochs, PPL 7.611012 |
| finemath_3plus/matched | 70%, 11.098047 epochs, PPL 7.533989 | 70%, 11.098047 epochs, PPL 11.439868 |
| finemath_3plus/target | 20%, 3.160974 epochs, PPL 4.461628 | 20%, 3.160974 epochs, PPL 5.202236 |

Wikipedia has a complete target grid. Its MATH-500 minimum is at 50% (7.937188 epochs), although 20% is only 0.007064 NLL worse. Its GSM8K minimum is at 20% (3.160974 epochs). FineMath has only 0%, 5%, 10%, 20% target measurements in this release; both scores continue improving through 20%, so that endpoint is the lowest observed value, not an identified optimum.

The rendered plot was inspected. It preserves all measured points, explicitly marks the partial FineMath tracks, and shows different target/proxy zero-fraction losses. The four remaining FineMath targets are absent from this frozen release; resumed training and subsequent evaluations must appear under their separate completion specification.

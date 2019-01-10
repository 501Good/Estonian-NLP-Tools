# Estonian-NLP-Tools

## Tokenizer

The tokenizer uses a bidirectional LSTM model trained on the [Estonian UD corpus v2.3](https://github.com/UniversalDependencies/UD_Estonian-EDT). 

The corpus was cut into the chunks of five sentences. Each chunk is then was transformed into a sequence of tags for each character: X - out of token, E - end of token, T - end of sentence. 
End of token and end of sentence tags are placed on the last character of a word/sentence. 

The model was trained until it showed no improvement in validation loss for five consecutive epochs (120 epochs in total).

The F1-score reported by the official UD evaluation script is shown in the table below:

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Words</th>
            <th>Sents</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>UDPipe</td>
            <td>99.95%</td>
            <td>91.72%	</td>
        </tr>
        <tr>
            <td>LSTM</td>
            <td>99.94%</td>
            <td>90.22%</td>
        </tr>
    </tbody>
</table>

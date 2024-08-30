def insert_span_ids(eval_df, spans_dataframe):
    eval_df["context.span_id"] = spans_dataframe.index[-1]
    eval_df.set_index("context.span_id", inplace=True)

    return eval_df



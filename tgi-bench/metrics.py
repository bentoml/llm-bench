import pandas as pd
import matplotlib.pyplot as plt 

metrics_df = pd.read_csv('../benchmark-openllm-user=100_time=900.0.csv')


def plot_metrics(metrics):
        # Prepare the data for plotting
        times = metrics['time_elapsed']
        total_requests = metrics['total_requests']
        requests_per_second = metrics['requests_per_second']
        response_latency = metrics['response_latency']
        response_tokens_per_second = metrics['response_tokens_per_second']
        gpu_utilization = metrics['gpu_utilization']
        cpu_utilization = metrics['cpu_utilization']

        # Create a 3x2 grid of subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('OpenLLM Benchmarking with Llama2-7b')

        # Plot each metric in its subplot
        axs[0, 0].plot(times, total_requests, label='Total Requests')
        axs[0, 0].set_title('Total Requests')
        axs[0, 0].set_xlabel('Time Elapsed (s)')
        axs[0, 0].set_ylabel('Total Requests')

        axs[0, 1].plot(times, requests_per_second, label='Requests/s')
        axs[0, 1].set_title('Requests per Second')
        axs[0, 1].set_xlabel('Time Elapsed (s)')
        axs[0, 1].set_ylabel('Requests/s')

        axs[0, 2].plot(times, response_latency, label='Response Latency')
        axs[0, 2].set_title('Response Latency')
        axs[0, 2].set_xlabel('Time Elapsed (s)')
        axs[0, 2].set_ylabel('Latency')

        axs[1, 0].plot(times, response_tokens_per_second, label='Response Tokens/s')
        axs[1, 0].set_title('Response Tokens per Second')
        axs[1, 0].set_xlabel('Time Elapsed (s)')
        axs[1, 0].set_ylabel('Tokens/s')

        axs[1, 1].plot(times, gpu_utilization, label='GPU Utilization')
        axs[1, 1].set_title('GPU Utilization')
        axs[1, 1].set_xlabel('Time Elapsed (s)')
        axs[1, 1].set_ylabel('Utilization (%)')

        axs[1, 2].plot(times, cpu_utilization, label='CPU Utilization')
        axs[1, 2].set_title('CPU Utilization')
        axs[1, 2].set_xlabel('Time Elapsed (s)')
        axs[1, 2].set_ylabel('Utilization (%)')

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../benchmark-metrics-openllm-user=100_time=900.png')
        plt.show()
        
plot_metrics(metrics_df)
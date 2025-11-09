import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from src.model import DiffuSDH


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler, dataset_name):
    # For performance, keep tensors on their original device (e.g., GPU).
    # The explicit .cpu() calls here were a bottleneck and are not necessary.
    
    # Inverse normalize to the final evaluation scale
    scaler_expanded = torch.tensor(scaler, device=target.device).unsqueeze(0).unsqueeze(2)
    mean_scaler_expanded = torch.tensor(mean_scaler, device=target.device).unsqueeze(0).unsqueeze(2)
    target_final_scale = target * scaler_expanded + mean_scaler_expanded
    forecast_final_scale = forecast * scaler_expanded + mean_scaler_expanded

    quantiles = np.arange(0.05, 1.0, 0.05)
    CRPS_sum = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast_final_scale, quantiles[i], dim=1)
        q_loss = quantile_loss(target_final_scale, q_pred, quantiles[i], eval_points)
        CRPS_sum += q_loss
    
    return CRPS_sum.item() / len(quantiles)




def evaluate(model: DiffuSDH, test_loader, device, probabilistic_eval=False, nsample=1, scaler=1, mean_scaler=0, results_save_path=None, dataset_name=None):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_evalpoint = []
        all_generated_samples = []

        # 优化：在循环外创建 scaler Tensors，避免重复创建
        # mean_scaler_expanded = torch.tensor(mean_scaler, device=device).unsqueeze(0).unsqueeze(2)
        scaler_tensor = torch.tensor(scaler, device=device).unsqueeze(0).unsqueeze(2)
        with tqdm(total=len(test_loader), desc="[评估]", ncols=100, unit="batch") as pbar:
            for batch_no, test_batch in enumerate(test_loader, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, _, _ = output
                
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_generated_samples.append(samples)

                # --- METRIC CALCULATION ON FINAL SCALE ---
                # 1. Inverse-normalize to the final evaluation scale
                # 优化：使用在循环外创建的 Tensors
                # median_pred_final = samples_median.values * scaler_expanded + mean_scaler_expanded
                # target_final = c_target * scaler_expanded + mean_scaler_expanded

                # 2. Calculate metrics directly on the final scale
                # --- 正确的评估流程：先反归一化，再计算指标 ---
                # 1. 反归一化预测值和真实值到原始尺度
                mean_scaler_expanded = torch.tensor(mean_scaler, device=device).unsqueeze(0).unsqueeze(2)
                scaler_expanded = scaler_tensor.to(c_target.device)
                
                pred_unscaled = samples_median.values * scaler_expanded + mean_scaler_expanded
                target_unscaled = c_target * scaler_expanded + mean_scaler_expanded

                # 2. 在原始尺度上计算 MSE 和 MAE
                mse_res = ((pred_unscaled - target_unscaled) * eval_points) ** 2
                mae_res = torch.abs((pred_unscaled - target_unscaled) * eval_points)

                # 3. 累加总误差
                mse_total += mse_res.sum().item()
                mae_total += mae_res.sum().item()
                evalpoints_total += eval_points.sum().item()

                pbar.set_postfix(
                    ordered_dict={
                        "RMSE": f"{np.sqrt(mse_total / evalpoints_total):.4f}",
                        "MAE": f"{mae_total / evalpoints_total:.4f}",
                        "MSE": f"{mse_total / evalpoints_total:.4f}",
                    },
                    refresh=True,
                )
                pbar.update(1)
                
        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_generated_samples = torch.cat(all_generated_samples, dim=0)

        final_rmse = np.sqrt(mse_total / evalpoints_total)
        final_mae = mae_total / evalpoints_total
        final_mse = mse_total / evalpoints_total

        print(f"\n--- 点预测评估结果 ({'对数尺度' if dataset_name == 'pre' else '物理尺度'}) ---")
        print(f"RMSE: {final_rmse:.4f}")
        print(f"MAE: {final_mae:.4f}")
        print(f"MSE: {final_mse:.4f}")

        if probabilistic_eval:
            print(f"\n--- 概率预测评估结果 ({'对数尺度' if dataset_name == 'pre' else '物理尺度'}) ---")
            total_q_loss_avg_quantile = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler, dataset_name
            )
            final_CRPS = total_q_loss_avg_quantile / evalpoints_total
            print(f"CRPS: {final_CRPS:.4f}")

            # For interval width, calculate on the final scale
            # 优化：复用已在正确设备上的 scaler_expanded 和 mean_scaler_expanded
            lower_bound = torch.quantile(all_generated_samples, 0.05, dim=1) 
            upper_bound = torch.quantile(all_generated_samples, 0.95, dim=1)

            # 优化：只创建一次 scaler Tensors
            scaler_expanded = torch.tensor(scaler, device=all_target.device).unsqueeze(0).unsqueeze(2)
            mean_scaler_expanded = torch.tensor(mean_scaler, device=all_target.device).unsqueeze(0).unsqueeze(2)

            lower_bound_unscaled = lower_bound * scaler_expanded + mean_scaler_expanded
            upper_bound_unscaled = upper_bound * scaler_expanded + mean_scaler_expanded
            # 优化：在 GPU 上完成计算，仅在最后将结果传输到 CPU
            interval_width = (upper_bound_unscaled - lower_bound_unscaled) * all_evalpoint
            avg_interval_width = (interval_width.sum() / all_evalpoint.sum()).item()
            print(f"平均 90% 预测区间宽度: {avg_interval_width:.4f}")

        if results_save_path:
            # 优化：复用已在正确设备上的 scaler_expanded 和 mean_scaler_expanded
            # Save results on the final scale
            target_final = all_target * scaler_expanded + mean_scaler_expanded
            median_pred_final = torch.quantile(all_generated_samples, 0.5, dim=1) * scaler_expanded + mean_scaler_expanded
            
            save_dict = {
                "ground_truth": target_final.cpu().numpy(),
                "median_pred": median_pred_final.cpu().numpy(),
                "eval_points": all_evalpoint.cpu().numpy()
            }

            if probabilistic_eval:
                lower_bound_final = torch.quantile(all_generated_samples, 0.05, dim=1) * scaler_expanded + mean_scaler_expanded
                upper_bound_final = torch.quantile(all_generated_samples, 0.95, dim=1) * scaler_expanded + mean_scaler_expanded
                
                save_dict["lower_bound"] = lower_bound_final.cpu().numpy()
                save_dict["upper_bound"] = upper_bound_final.cpu().numpy()

            np.savez_compressed(results_save_path, **save_dict)
            print(f"\n评估结果已保存至: {results_save_path}")

        if probabilistic_eval:
            return final_rmse, final_mae, final_mse, final_CRPS, avg_interval_width
        
        return final_rmse, final_mae, final_mse
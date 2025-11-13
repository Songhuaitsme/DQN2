
import config

class Server:
    """
    单个服务器模型 (Single Server Model)
    (*** 已修改为支持“并发任务”，但无 80% 硬限制 ***)
    """

    def __init__(self, id, rack_id):
        self.id = id
        self.rack_id = rack_id
        self.total_cpu = config.CPU_PER_SERVER  # (e.g., 24 CPU)
        self.used_cpu = 0

        # --- (*** 关键修改 1: 变更为任务列表 ***) ---
        self.current_tasks = []

        # 服务器状态 (Server State)
        self.utilization = 0.0
        self.power = config.P_IDLE

    def _update_utilization(self):
        """ Eq. 4: u_k = c_using / c_total """
        self.utilization = self.used_cpu / self.total_cpu

    def _update_power(self):
        """ Eq. 5: p_k = p_idle + (p_full - p_idle) * u_k """
        self.power = config.P_IDLE + (config.P_FULL - config.P_IDLE) * self.utilization

    def update_server_state(self):
        """ 更新服务器的利用率和功耗 (Update server utilization and power) """
        self._update_utilization()
        self._update_power()

    # --- (*** 关键修改 2: can_assign_task 只检查 100% 物理容量 ***) ---
    def can_assign_task(self, task):
        """
        检查服务器是否可以分配此任务
        (Check if server can be assigned this task)

        唯一的约束是服务器的物理 CPU 上限 (e.g., 24)
        """
        has_capacity = (self.total_cpu - self.used_cpu) >= task.cpu_needed
        return has_capacity

    # --- (*** 关键修改 3: assign_task 追加到列表 ***) ---
    def assign_task(self, task):
        """
        分配任务到此服务器 (Assign task to this server)
        """
        if not self.can_assign_task(task):
            return False

        self.used_cpu += task.cpu_needed

        # 将任务及其持续时间打包为一个字典，存入列表
        task_info = {
            "task_obj": task,
            "remaining_duration": task.duration
        }
        self.current_tasks.append(task_info)

        self.update_server_state()
        return True

    # --- (*** 关键修改 4: release_task 释放特定任务 ***) ---
    def release_task(self, task_info_to_release):
        """ 任务完成，释放资源 (Task complete, release resources) """
        self.used_cpu -= task_info_to_release["task_obj"].cpu_needed
        self.current_tasks.remove(task_info_to_release)  # 从列表中移除
        self.update_server_state()

    # --- (*** 关键修改 5: step 迭代所有并发任务 ***) ---
    def step(self):
        """ 模拟一个时间步的前进 (Simulate one time step progression) """

        if not self.current_tasks:
            return

        # 必须迭代列表的 *副本* (list(...))
        # 因为我们将在循环内部修改原始列表 (self.release_task)
        for task_info in list(self.current_tasks):
            task_info["remaining_duration"] -= 1
            if task_info["remaining_duration"] == 0:
                # 此特定任务已完成
                self.release_task(task_info)


class Rack:
    """
    机架模型 (Rack model)
    (无需修改)
    """

    def __init__(self, id):
        self.id = id
        self.servers = [
            Server(id=(id * config.SERVERS_PER_RACK + i), rack_id=id)
            for i in range(config.SERVERS_PER_RACK)
        ]

    def get_total_power(self):
        """ 计算此机架的总功耗 (Calculate total power for this rack) """
        return sum(s.power for s in self.servers)
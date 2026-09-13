package main

import (
	"os"

	kubeattention "github.com/softcane/KubeAttention/pkg/scheduler"
	"k8s.io/component-base/cli"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
)

func main() {
	command := app.NewSchedulerCommand(
		app.WithPlugin(kubeattention.PluginName, kubeattention.New),
	)
	os.Exit(cli.Run(command))
}
